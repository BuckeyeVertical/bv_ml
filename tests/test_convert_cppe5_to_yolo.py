import io
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image

import convert_cppe5_to_yolo as converter


class Cppe5ConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "v002"
        self.source.mkdir()
        self.names = ["Coverall", "Mask"]

    def row(self, image_id, color, boxes=None, categories=None):
        data = io.BytesIO()
        Image.new("RGB", (200, 100), color).save(data, format="PNG")
        return {
            "image_id": image_id, "width": 200, "height": 100,
            "image": {"bytes": data.getvalue(), "path": f"{image_id}.png"},
            "objects": {"bbox": [[20., 10., 40., 20.]] if boxes is None else boxes,
                        "category": [1] if categories is None else categories},
        }

    def write(self, train=None):
        metadata = {b"huggingface": json.dumps({"info": {"features": {"objects": {
            "feature": {"category": {"names": self.names}},
        }}}}).encode()}
        rows = [self.row(i, color) for i, color in enumerate(("red", "green", "blue", "white"))]
        for split, values in (("train", rows if train is None else train),
                              ("test", [self.row(10, "black")])):
            pq.write_table(pa.Table.from_pylist(values).replace_schema_metadata(metadata),
                           self.source / f"{split}-00000-of-00001.parquet")

    def test_export_labels_image_pairs_and_reproducible_independent_splits(self):
        self.write()
        first = converter.convert(self.source, val_fraction=0.25)
        second = converter.convert(self.source, self.root / "second", val_fraction=0.25)
        self.assertEqual(first, second)
        self.assertEqual({s: v["images"] for s, v in first["splits"].items()}, {"train": 3, "val": 1, "test": 1})
        output = self.source / "yolo"
        config = yaml.safe_load((output / "data.yaml").read_text())
        self.assertNotIn("path", config)
        self.assertEqual(config["names"], dict(enumerate(self.names)))
        for split in ("train", "val", "test"):
            images = sorted((output / config[split]).glob("*.png"))
            labels = sorted((output / "labels" / split).glob("*.txt"))
            self.assertEqual([p.stem for p in images], [p.stem for p in labels])
            for label in labels:
                self.assertEqual(label.read_text().split(), ["1", "0.2000000000", "0.2000000000", "0.2000000000", "0.2000000000"])
        self.assertTrue(all((self.source / record["source_file"]).is_file() for record in first["images"]))

    def test_deduplicates_before_splitting_and_prioritizes_test(self):
        self.write([self.row(1, "black"), self.row(2, "red"), self.row(3, "red"),
                    self.row(4, "blue"), self.row(5, "green")])
        report = converter.convert(self.source)
        self.assertEqual(len(report["duplicates_removed"]), 2)
        self.assertEqual(len({r["sha256"] for r in report["images"]}), 4)
        self.assertEqual([r["image_id"] for r in report["images"] if r["split"] == "test"], [10])

    def test_jpeg_bytes_with_png_source_name_get_correct_extension(self):
        row = self.row(1, "red")
        encoded = io.BytesIO()
        Image.new("RGB", (200, 100), "red").save(encoded, format="JPEG")
        row["image"]["bytes"] = encoded.getvalue()
        self.write([row, self.row(2, "blue"), self.row(3, "green")])
        report = converter.convert(self.source)
        record = next(r for r in report["images"] if r["image_id"] == 1)
        path = self.source / "yolo" / record["file"]
        self.assertEqual(path.suffix, ".jpg")
        self.assertEqual(path.read_bytes(), encoded.getvalue())

    def test_clips_boxes_and_reports_skipped_boxes(self):
        row = self.row(1, "red", [[-20., -10., 60., 30.], [300., 0., 10., 10.]], [0, 1])
        text, issues, counts = converter._labels(row, self.names)
        self.assertEqual(text.split(), ["0", "0.1000000000", "0.1000000000", "0.2000000000", "0.2000000000"])
        self.assertEqual(len(issues), 2)
        self.assertEqual(counts, {"Coverall": 1})

    def test_jpeg_missing_end_marker_is_exported_as_png(self):
        row = self.row(1, "red")
        encoded = io.BytesIO()
        Image.new("RGB", (200, 100), "red").save(encoded, format="JPEG")
        row["image"]["bytes"] = encoded.getvalue()[:-2]
        self.write([row, self.row(2, "blue"), self.row(3, "green")])
        report = converter.convert(self.source)
        self.assertEqual(len(report["image_repairs"]), 1)
        record = next(r for r in report["images"] if r["image_id"] == 1)
        path = self.source / "yolo" / record["file"]
        self.assertEqual(path.suffix, ".png")
        with Image.open(path) as image:
            image.verify()

    def test_bad_class_aborts_without_publishing_partial_output(self):
        row = self.row(1, "red", categories=[99])
        self.write([row, self.row(2, "blue")])
        with self.assertRaisesRegex(ValueError, "invalid category"):
            converter.convert(self.source)
        self.assertFalse((self.source / "yolo").exists())
        self.assertEqual(list(self.source.glob(".cppe5-yolo-*")), [])

    def test_empty_labels_for_background_and_dimension_validation(self):
        row = self.row(1, "red", [], [])
        self.assertEqual(converter._labels(row, self.names)[0], "")
        row["width"] = 1
        self.write([row, self.row(2, "blue")])
        with self.assertRaisesRegex(ValueError, "dimensions disagree"):
            converter.convert(self.source)
        self.assertFalse((self.source / "yolo").exists())

    def test_existing_outputs_and_invalid_fractions_are_rejected(self):
        self.write()
        for value in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                converter.convert(self.source, val_fraction=value)
        (self.source / "yolo").mkdir()
        marker = self.source / "yolo" / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            converter.convert(self.source)
        self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
