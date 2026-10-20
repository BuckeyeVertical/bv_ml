import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

import prepare_datasets


class DatasetConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dataset tests ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.version = self.root / "v1.example"
        self.yolo = self.version / "yolo"
        self.yolo.mkdir(parents=True)
        self.config = {"train": "images/train", "val": "images/val", "names": ["person", "car"]}
        self.write_config()
        self.add_image("train", "a.png", "0 0.5 0.5 0.2 0.4\n")
        self.add_image("val", "a.png", "1 0.5 0.5 0.3 0.3\n")

    def write_config(self):
        (self.yolo / "data.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def add_image(self, split, name, label=None):
        image = self.yolo / "images" / split / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 100), "white").save(image)
        if label is not None:
            path = (self.yolo / "labels" / split / name).with_suffix(".txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(label, encoding="utf-8")
        return image

    def convert(self, version=None):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = prepare_datasets.convert_yolo_to_coco(self.version if version is None else version)
        return result, output.getvalue()

    def annotations(self, output, split="train"):
        return json.loads((output / f"annotations/instances_{split}.json").read_text())

    def manifest(self, output):
        return json.loads((output / "manifest.json").read_text())

    def clone(self, name):
        return Path(shutil.copytree(self.version, self.root / name, symlinks=True))

    def assert_no_partial_output(self, version=None):
        version = self.version if version is None else version
        self.assertFalse((version / "coco").exists())
        self.assertEqual(list(version.glob(".coco-*")), [])

    def test_coordinates_categories_and_references(self):
        self.add_image("train", "nested/a.png", "1 0.5 0.5 0.5 0.5\n")
        output, _ = self.convert()
        coco = self.annotations(output)
        self.assertEqual([image["file_name"] for image in coco["images"]], ["a.png", "nested/a.png"])
        self.assertEqual(coco["images"][0], {"id": 1, "file_name": "a.png", "width": 200, "height": 100})
        for actual, expected in zip(coco["annotations"][0]["bbox"], [80, 30, 40, 40]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(coco["annotations"][0]["area"], 1600)
        self.assertEqual([category["id"] for category in coco["categories"]], [1, 2])
        self.assertEqual([category["name"] for category in coco["categories"]], ["person", "car"])
        self.assertEqual(self.annotations(output, "val")["annotations"][0]["category_id"], 2)
        self.assertEqual(coco["categories"], self.annotations(output, "val")["categories"])
        self.assertEqual([annotation["id"] for annotation in coco["annotations"]], [1, 2])
        for annotation in coco["annotations"]:
            self.assertIn(annotation["image_id"], {image["id"] for image in coco["images"]})
            self.assertIn(annotation["category_id"], {category["id"] for category in coco["categories"]})
            self.assertEqual(annotation["iscrowd"], 0)
        self.assertIn("info", coco)
        self.assertIn("licenses", coco)

    def test_images_are_relative_links_without_copies(self):
        output, _ = self.convert()
        for split in ("train", "val"):
            link = output / "images" / split
            self.assertTrue(link.is_symlink())
            self.assertFalse(os.path.isabs(os.readlink(link)))
            self.assertEqual(link.resolve(), self.yolo / "images" / split)
            self.assertEqual((link / "a.png").stat().st_ino, (self.yolo / "images" / split / "a.png").stat().st_ino)
        self.assertFalse((output / "annotations/instances_test.json").exists())

    def test_exif_orientation_controls_dimensions_and_boxes_without_changing_sources(self):
        sources = {}
        for suffix in ("jpg", "png", "tiff"):
            for orientation in range(1, 9):
                image = self.add_image("train", f"orientation-{orientation}.{suffix}", "0 0.25 0.5 0.25 0.5\n")
                exif = Image.Exif()
                exif[274] = orientation
                Image.new("RGB", (200, 100), "white").save(image, exif=exif)
                sources[image] = image.read_bytes()

        output, _ = self.convert()
        coco = self.annotations(output)
        images = {image["file_name"]: image for image in coco["images"]}
        annotations = {annotation["image_id"]: annotation for annotation in coco["annotations"]}
        for source in sources:
            orientation = int(source.stem.split("-")[1])
            with self.subTest(file=source.name):
                image = images[source.name]
                expected_size = (100, 200) if orientation >= 5 else (200, 100)
                expected_bbox = [12.5, 50, 25, 100] if orientation >= 5 else [25, 25, 50, 50]
                self.assertEqual((image["width"], image["height"]), expected_size)
                annotation = annotations[image["id"]]
                self.assertEqual(annotation["bbox"], expected_bbox)
                self.assertEqual(annotation["area"], 2500)
        for image, original in sources.items():
            self.assertEqual(image.read_bytes(), original)
        _, report = self.convert()
        self.assertIn("already up to date", report)

    def test_visdrone_layout_and_optional_test_split(self):
        self.config.update(train="VisDrone2019-DET-train/images", val="VisDrone2019-DET-val/images", test="VisDrone2019-DET-test-dev/images")
        self.write_config()
        for split, folder in [("train", "train"), ("val", "val"), ("test", "test-dev")]:
            target = self.yolo / f"VisDrone2019-DET-{folder}"
            shutil.copytree(self.yolo / "images/train", target / "images")
            shutil.copytree(self.yolo / "labels/train", target / "labels")
        output, _ = self.convert()
        self.assertEqual(os.readlink(output / "images/train"), "../../yolo/VisDrone2019-DET-train/images")
        self.assertEqual(len(self.annotations(output, "test")["annotations"]), 1)

    def test_yaml_path_and_class_mapping(self):
        (self.yolo / "export").mkdir()
        shutil.move(self.yolo / "images", self.yolo / "export/images")
        shutil.move(self.yolo / "labels", self.yolo / "export/labels")
        self.config.update(path="export", names={"1": "car", "0": "person"}, nc=2)
        self.write_config()
        output, _ = self.convert()
        self.assertEqual((output / "images/train").resolve(), self.yolo / "export/images/train")
        self.assertEqual(self.annotations(output)["categories"][0]["name"], "person")

    def test_missing_and_empty_labels_are_backgrounds(self):
        self.add_image("train", "b.png")
        self.add_image("train", "c.png", "\n \n")
        output, report = self.convert()
        summary = self.manifest(output)["summary"]["train"]
        self.assertEqual(summary["images"], 3)
        self.assertEqual(summary["annotations"], 1)
        self.assertEqual(summary["background_images"], 2)
        self.assertEqual(summary["missing_labels"], 1)
        self.assertEqual(summary["empty_labels"], 1)
        self.assertEqual(len(self.annotations(output)["images"]), 3)
        self.assertIn("1 missing / 1 empty", report)

    def test_zero_area_boxes_are_skipped_with_line_numbers(self):
        label = self.yolo / "labels/train/a.txt"
        label.write_text("0 0.5 0.5 0.2 0\n0 0.5 0.5 0.2 0.4\n")
        source = label.read_bytes()
        output, report = self.convert()
        manifest = self.manifest(output)
        self.assertEqual(manifest["summary"]["train"]["skipped_zero_area_boxes"], 1)
        self.assertEqual(manifest["issues"], [{"file": "yolo/labels/train/a.txt", "line": 1, "reason": "zero-area box"}])
        self.assertIn(f"{label}:1: skipped zero-area box", report)
        self.assertEqual(len(self.annotations(output)["annotations"]), 1)
        self.assertEqual(label.read_bytes(), source)

    def test_rounding_overflow_is_clamped(self):
        (self.yolo / "labels/train/a.txt").write_text("0 0.09999975 0.5 0.2 0.4\n0 0.90000025 0.5 0.2 0.4\n")
        output, _ = self.convert()
        self.assertEqual(self.manifest(output)["summary"]["train"]["clamped_boxes"], 2)
        for annotation in self.annotations(output)["annotations"]:
            left, top, width, height = annotation["bbox"]
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(left + width, 200)
            self.assertLessEqual(top + height, 100)

    def test_invalid_rows_fail_with_location_and_no_partial_output(self):
        rows = [
            "0 0.5 0.5 0.1", "0 0.5 0.5 0.1 0.1 0.1", "x 0.5 0.5 0.1 0.1",
            "0 nan 0.5 0.1 0.1", "0 0.5 inf 0.1 0.1", "2 0.5 0.5 0.1 0.1",
            "0.5 0.5 0.5 0.1 0.1", "0 0.5 0.5 -0.1 0.1", "0 0.5 0.5 0.1 -0.1",
            "0 0.01 0.5 0.2 0.1",
        ]
        label = self.yolo / "labels/train/a.txt"
        for row in rows:
            with self.subTest(row=row):
                label.write_text(row + "\n")
                with self.assertRaisesRegex(ValueError, "a.txt:1:"):
                    self.convert()
                self.assert_no_partial_output()

    def test_unreadable_images_fail_without_publishing(self):
        (self.yolo / "images/train/a.png").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "Unreadable image"):
            self.convert()
        self.assert_no_partial_output()

    def test_rerun_is_a_no_op_and_output_is_deterministic(self):
        output, _ = self.convert()
        original = {path.relative_to(output): (path.read_bytes(), path.stat().st_mtime_ns) for path in output.rglob("*.json")}
        result, report = self.convert()
        self.assertEqual(result, output)
        self.assertIn("already up to date", report)
        self.assertEqual(original, {path.relative_to(output): (path.read_bytes(), path.stat().st_mtime_ns) for path in output.rglob("*.json")})
        other = self.root / "v2.copy"
        shutil.copytree(self.yolo, other / "yolo")
        converted, _ = self.convert(other)
        self.assertEqual({relative: value[0] for relative, value in original.items()}, {path.relative_to(converted): path.read_bytes() for path in converted.rglob("*.json")})

    def test_links_and_manifest_survive_relocation(self):
        self.convert()
        moved = self.root / "v1.moved"
        shutil.move(self.version, moved)
        output, report = self.convert(moved)
        self.assertEqual((output / "images/train").resolve(), moved / "yolo/images/train")
        self.assertIn("already up to date", report)

    def test_old_conversion_version_is_preserved_and_requires_regeneration(self):
        with patch.object(prepare_datasets, "_FORMAT_VERSION", 1):
            output, _ = self.convert()
        original = {path: path.read_bytes() for path in output.rglob("*.json")}
        with self.assertRaisesRegex(ValueError, "conversion version 1.*Move the generated coco directory aside and rerun"):
            self.convert()
        self.assertEqual({path: path.read_bytes() for path in original}, original)

    def test_source_changes_require_new_versions_and_preserve_output(self):
        self.convert()
        for change in ("labels", "yaml", "pixels", "new_image", "removed_image"):
            with self.subTest(change=change):
                version = self.clone(f"v2.{change}")
                source = version / "yolo"
                if change == "labels":
                    (source / "labels/train/a.txt").write_text("0 0.5 0.5 0.3 0.3\n")
                elif change == "yaml":
                    config = yaml.safe_load((source / "data.yaml").read_text())
                    config["names"][0] = "changed"
                    (source / "data.yaml").write_text(yaml.safe_dump(config))
                elif change == "pixels":
                    Image.new("RGB", (200, 100), "black").save(source / "images/train/a.png")
                elif change == "new_image":
                    shutil.copy2(source / "images/train/a.png", source / "images/train/b.png")
                else:
                    (source / "images/train/a.png").unlink()
                    shutil.copy2(source / "images/val/a.png", source / "images/train/b.png")
                original = (version / "coco/manifest.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "create a new dataset version"):
                    self.convert(version)
                self.assertEqual((version / "coco/manifest.json").read_bytes(), original)

    def test_new_label_for_background_counts_as_source_change(self):
        self.add_image("train", "b.png")
        self.convert()
        (self.yolo / "labels/train/b.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        with self.assertRaisesRegex(ValueError, "create a new dataset version"):
            self.convert()

    def test_training_caches_and_file_timestamps_do_not_change_fingerprint(self):
        self.convert()
        (self.yolo / "labels/train.cache").write_bytes(b"training cache")
        (self.yolo / "images/train/a.png").touch()
        _, report = self.convert()
        self.assertIn("already up to date", report)

    def test_damaged_output_is_preserved_and_reported(self):
        self.convert()
        for damage in ("json", "link", "manifest"):
            with self.subTest(damage=damage):
                version = self.clone(f"v2.{damage}")
                output = version / "coco"
                if damage == "json":
                    (output / "annotations/instances_train.json").write_text("{}")
                elif damage == "link":
                    (output / "images/train").unlink()
                    (output / "images/train").symlink_to("../../yolo/images/val")
                else:
                    (output / "manifest.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "unexpected or damaged"):
                    self.convert(version)
                self.assertTrue(output.is_dir())

    def test_unrecognized_existing_directory_is_not_overwritten(self):
        output = self.version / "coco"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "unexpected or damaged"):
            self.convert()
        self.assertEqual(marker.read_text(), "keep me")

    def test_malformed_manifest_reports_an_actionable_error(self):
        output, _ = self.convert()
        original = self.manifest(output)
        for field, value in [("annotations", []), ("summary", {"train": {}, "val": {}}), ("issues", None)]:
            with self.subTest(field=field):
                manifest = {**original, field: value}
                (output / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "unexpected or damaged"):
                    self.convert()

    def test_failure_after_writing_annotations_cleans_staging(self):
        original = prepare_datasets._write_json

        def fail_second(path, value):
            if path.name == "instances_val.json":
                raise OSError("simulated disk failure")
            original(path, value)

        with (
            patch.object(prepare_datasets, "_write_json", side_effect=fail_second),
            self.assertRaisesRegex(OSError, "simulated disk failure"),
        ):
            self.convert()
        self.assert_no_partial_output()

    def test_source_changes_during_conversion_abort_publication(self):
        original = prepare_datasets._write_json

        def edit_source(path, value):
            original(path, value)
            if path.name == "instances_train.json":
                (self.yolo / "labels/train/a.txt").write_text("0 0.5 0.5 0.3 0.3\n")

        with (
            patch.object(prepare_datasets, "_write_json", side_effect=edit_source),
            self.assertRaisesRegex(ValueError, "source changed during conversion"),
        ):
            self.convert()
        self.assert_no_partial_output()

    def test_concurrent_conversions_only_build_once(self):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.object(prepare_datasets, "_convert_split", wraps=prepare_datasets._convert_split) as convert,
            ThreadPoolExecutor(max_workers=3) as pool,
        ):
            results = list(pool.map(prepare_datasets.convert_yolo_to_coco, [self.version] * 3))
        self.assertEqual(results, [self.version / "coco"] * 3)
        self.assertEqual(convert.call_count, 2)
        self.assertEqual(list(self.version.glob(".coco-*")), [])

    def test_unsupported_split_inputs_and_invalid_class_maps_fail(self):
        configs = [
            {**self.config, "train": ["images/train"]},
            {**self.config, "train": "train.txt"},
            {**self.config, "names": {1: "person"}},
            {**self.config, "nc": 99},
            {**self.config, "names": []},
        ]
        for config in configs:
            with self.subTest(config=config):
                (self.yolo / "data.yaml").write_text(yaml.safe_dump(config))
                with self.assertRaises((TypeError, ValueError)):
                    self.convert()
                self.assert_no_partial_output()

    def test_consolidated_cli_converts_and_archives_the_dataset(self):
        data_root = self.root / "data"
        data_root.mkdir()
        shutil.move(self.version, data_root / "v001.example")
        report = self.root / "summary.json"
        with contextlib.redirect_stdout(io.StringIO()):
            status = prepare_datasets.main(["--run", "--data-root", str(data_root), "--report", str(report)])
        self.assertEqual(status, 0)
        self.assertTrue((data_root / "v001.example/coco/manifest.json").is_file())
        self.assertTrue((data_root / "archives/v001.example.tar").is_file())
        self.assertEqual(json.loads(report.read_text())["prepared"][0]["version"], "v001.example")


if __name__ == "__main__":
    unittest.main()
