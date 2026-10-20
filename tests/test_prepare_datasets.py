import contextlib
import io
import json
import os
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import prepare_datasets


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="preparation tests '")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        root_patch = patch.object(prepare_datasets, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def make_dataset(self, name, *, raw=False, parent=None):
        version = (self.data if parent is None else parent) / name
        yolo = version if raw else version / "yolo"
        images = yolo / "images/train"
        images.mkdir(parents=True)
        Image.new("RGB", (20, 10), "white").save(images / "a.png")
        (yolo / "data.yaml").write_text("train: images/train\nnames: [person]\n")
        return version

    def make_zip(self, filename, *, raw=False, wrapper=None, coco=False):
        source = self.make_dataset("contents", raw=raw, parent=self.root / "ZIP sources" / filename)
        if coco:
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_datasets.convert_yolo_to_coco(source)
        destination = self.data / filename
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in source.rglob("*"):
                relative = path.relative_to(source)
                member = (Path(wrapper) / relative if wrapper else relative).as_posix()
                if path.is_symlink():
                    info = zipfile.ZipInfo(member)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    archive.writestr(info, os.readlink(path))
                elif path.is_file():
                    archive.write(path, arcname=member)
            archive.writestr("__MACOSX/._metadata", "ignored")
        return destination

    def test_names_are_normalized_and_raw_yolo_is_wrapped_before_conversion(self):
        existing = self.make_dataset("v010.keep")
        legacy = self.make_dataset("v1.legacy")
        unversioned = self.make_dataset("alpha")
        raw = self.make_dataset("bravo", raw=True)
        already_numbered_raw = self.make_dataset("v005.raw", raw=True)
        image = (raw / "images/train/a.png").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_datasets.convert_yolo_to_coco(legacy)
            summary = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual([item["version"] for item in summary["prepared"]], [
            "v001.legacy", "v005.raw", "v010.keep", "v011.alpha", "v012.bravo",
        ])
        self.assertEqual(summary["skipped"], [])
        self.assertTrue(existing.is_dir())
        for original in (legacy, unversioned, raw):
            self.assertFalse(original.exists())
        self.assertEqual((self.data / "v012.bravo/yolo/images/train/a.png").read_bytes(), image)
        self.assertTrue((self.data / "v012.bravo/yolo/data.yaml").is_file())
        self.assertFalse((already_numbered_raw / "data.yaml").exists())
        self.assertTrue((already_numbered_raw / "yolo/data.yaml").is_file())
        self.assertEqual(set(summary["wrapped"]), {"v005.raw", "v012.bravo"})
        self.assertEqual((self.data / "v001.legacy/coco/images/train").resolve(), self.data / "v001.legacy/yolo/images/train")
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual(repeated["prepared"], summary["prepared"])
        self.assertEqual(repeated["renamed"], [])
        self.assertEqual(repeated["wrapped"], [])

    def test_naming_collisions_never_overwrite_existing_inputs(self):
        existing = self.make_dataset("v001.same")
        legacy = self.make_dataset("v1.same")
        (existing / "identity.txt").write_text("existing")
        (legacy / "identity.txt").write_text("legacy")
        blocker = self.data / "v002.same"
        blocker.write_text("preserve this file")
        with contextlib.redirect_stdout(io.StringIO()):
            summary = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual([item["version"] for item in summary["prepared"]], ["v001.same", "v003.same"])
        self.assertEqual((existing / "identity.txt").read_text(), "existing")
        self.assertEqual((self.data / "v003.same/identity.txt").read_text(), "legacy")
        self.assertEqual(blocker.read_text(), "preserve this file")
        self.assertEqual(summary["renamed"], [{"from": "v1.same", "to": "v003.same"}])

    def test_failed_raw_wrap_restores_original_directory_and_continues(self):
        raw = self.make_dataset("raw", raw=True)
        good = self.make_dataset("v010.good")
        original = (raw / "data.yaml").read_bytes()
        real_rename = Path.rename

        def fail_publish(path, target):
            if path.name.startswith(".prepare-layout-"):
                raise OSError("cannot publish wrapped dataset")
            return real_rename(path, target)

        with patch.object(Path, "rename", fail_publish), contextlib.redirect_stdout(io.StringIO()):
            summary = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual([item["version"] for item in summary["prepared"]], [good.name])
        self.assertEqual(summary["skipped"][0]["stage"], "normalize")
        self.assertEqual((raw / "data.yaml").read_bytes(), original)
        self.assertFalse((self.data / "v011.raw").exists())
        self.assertEqual(list(self.data.glob(".prepare-layout-*")), [])

    def test_zips_support_flat_wrapped_and_raw_layouts_without_duplicate_imports(self):
        self.make_dataset("v002.keep")
        flat = self.make_zip("flat.zip", coco=True)
        raw = self.make_zip("raw.zip", raw=True)
        wrapped = self.make_zip("v7.wrapped.ZIP", raw=True, wrapper="outer/dataset")
        originals = {path: path.read_bytes() for path in (flat, raw, wrapped)}
        with contextlib.redirect_stdout(io.StringIO()):
            summary = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual({item["source"]: item["version"] for item in summary["imported"]}, {
            "flat.zip": "v008.flat", "raw.zip": "v009.raw", "v7.wrapped.ZIP": "v007.wrapped",
        })
        self.assertEqual(summary["skipped"], [])
        self.assertTrue((self.data / "v008.flat/coco/images/train").is_symlink())
        self.assertTrue((self.data / "v009.raw/yolo/data.yaml").is_file())
        self.assertTrue((self.data / "v007.wrapped/yolo/data.yaml").is_file())
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual(repeated["prepared"], summary["prepared"])
        self.assertEqual(repeated["imported"], [])
        self.assertEqual(len(json.loads((self.data / ".prepared-zips.json").read_text())), 3)
        self.assertEqual(list(self.data.glob(".prepare-zip-*")), [])

    def test_changed_zip_creates_a_new_version_and_preserves_the_previous_one(self):
        source = self.make_zip("upload.zip", raw=True)
        with contextlib.redirect_stdout(io.StringIO()):
            first = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual(first["imported"], [{"source": "upload.zip", "version": "v001.upload"}])
        with zipfile.ZipFile(source, "a") as archive:
            archive.writestr("new.txt", "new upload")
        with contextlib.redirect_stdout(io.StringIO()):
            second = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual(second["imported"], [{"source": "upload.zip", "version": "v002.upload"}])
        self.assertFalse((self.data / "v001.upload/yolo/new.txt").exists())
        self.assertEqual((self.data / "v002.upload/yolo/new.txt").read_text(), "new upload")
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual(repeated["imported"], [])
        self.assertEqual(len(repeated["prepared"]), 2)

    def test_invalid_and_escaping_zip_entries_are_skipped_without_writing_outside_staging(self):
        self.make_zip("good.zip", raw=True)
        (self.data / "broken.zip").write_bytes(b"invalid zip")
        outside = self.root / "outside.txt"
        for filename, member in (("relative.zip", "../../outside.txt"), ("absolute.zip", str(outside))):
            with zipfile.ZipFile(self.data / filename, "w") as archive:
                archive.writestr(member, "must not escape")
        with zipfile.ZipFile(self.data / "symlink.zip", "w") as archive:
            info = zipfile.ZipInfo("bad-link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, str(outside))
        with contextlib.redirect_stdout(io.StringIO()):
            summary = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual([item["version"] for item in summary["prepared"]], ["v001.good"])
        self.assertEqual({item["version"] for item in summary["skipped"]}, {"broken.zip", "relative.zip", "absolute.zip", "symlink.zip"})
        self.assertTrue(all(item["stage"] == "extract" for item in summary["skipped"]))
        self.assertFalse(outside.exists())
        self.assertFalse((self.data / "outside.txt").exists())
        self.assertEqual(list(self.data.glob(".prepare-zip-*")), [])

    def test_zip_only_failures_are_included_in_the_summary(self):
        (self.data / "broken.zip").write_bytes(b"invalid zip")
        report = self.root / "summary.json"
        with contextlib.redirect_stdout(io.StringIO()):
            status = prepare_datasets.main(["--run", "--report", str(report)])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(report.read_text())["skipped"][0]["stage"], "extract")

    def test_concurrent_preparation_does_not_assign_duplicate_names_or_import_zips_twice(self):
        self.make_dataset("folder", raw=True)
        self.make_zip("upload.zip", raw=True)
        with contextlib.redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(prepare_datasets.prepare_datasets, [self.data] * 3))
        self.assertTrue(all(len(result["prepared"]) == 2 and not result["skipped"] for result in results))
        self.assertEqual(sum(len(result["imported"]) for result in results), 1)
        self.assertEqual({path.name for path in self.data.glob("v*")}, {"v001.folder", "v002.upload"})

    def test_version_directory_symlinks_keep_their_normalized_archive_names(self):
        shared = self.make_dataset("original", parent=self.root / "shared")
        source = self.data / "v1.alias"
        source.symlink_to(shared, target_is_directory=True)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual(summary["skipped"], [])
        self.assertTrue((self.data / "v001.alias").is_symlink())
        self.assertTrue(shared.is_dir())
        archive_path = self.data / "archives/v001.alias.tar"
        self.assertEqual(Path(summary["prepared"][0]["archive"]), archive_path)
        with tarfile.open(archive_path) as archive:
            self.assertIn("v001.alias/yolo/data.yaml", archive.getnames())

    def test_mixed_versions_continue_after_conversion_failure_and_refresh_archives(self):
        first = self.make_dataset("v001.first")
        broken = self.data / "v002.broken"
        broken.mkdir()
        last = self.make_dataset("v010.last")
        (self.data / "notes").mkdir()
        (self.data / "v999.file").write_text("not a dataset")
        with contextlib.redirect_stdout(io.StringIO()):
            old_archive = prepare_datasets.archive_dataset(first)
        with tarfile.open(old_archive) as archive:
            self.assertNotIn(f"{first.name}/coco/manifest.json", archive.getnames())

        with contextlib.redirect_stdout(io.StringIO()) as log:
            summary = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual([item["version"] for item in summary["prepared"]], [first.name, last.name])
        self.assertEqual([item["version"] for item in summary["skipped"]], [broken.name])
        self.assertEqual(summary["skipped"][0]["stage"], "coco")
        self.assertIn("data.yaml", summary["skipped"][0]["error"])
        self.assertIn("SKIPPED v002.broken", log.getvalue())
        self.assertFalse((self.data / "archives" / f"{broken.name}.tar").exists())
        for item in summary["prepared"]:
            with tarfile.open(item["archive"]) as archive:
                manifest = archive.extractfile(f"{item['version']}/coco/manifest.json").read()
                self.assertEqual(json.loads(manifest)["format"], "coco-detection")
                self.assertTrue(archive.getmember(f"{item['version']}/coco/images/train").issym())

        with contextlib.redirect_stdout(io.StringIO()) as repeated_log:
            repeated = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual(repeated, summary)
        self.assertIn("COCO already up to date", repeated_log.getvalue())

    def test_archive_failure_preserves_previous_archive_and_continues(self):
        first = self.make_dataset("v001.first")
        last = self.make_dataset("v002.last")
        with contextlib.redirect_stdout(io.StringIO()):
            old_archive = prepare_datasets.archive_dataset(first)
        original = old_archive.read_bytes()
        real_archive = prepare_datasets.archive_dataset

        def archive_or_fail(version, *, force):
            if version == first:
                raise OSError("storage full")
            return real_archive(version, force=force)

        with (
            patch.object(prepare_datasets, "archive_dataset", side_effect=archive_or_fail),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            summary = prepare_datasets.prepare_datasets(self.data)
        self.assertEqual([item["version"] for item in summary["prepared"]], [last.name])
        self.assertEqual(summary["skipped"], [{"version": first.name, "stage": "archive", "error": "OSError: storage full"}])
        self.assertEqual(old_archive.read_bytes(), original)

    def test_default_command_submits_cpu_job_without_preparing_data_on_login_node(self):
        self.make_dataset("v001.example")
        result = subprocess.CompletedProcess(["sbatch"], 0, "Submitted batch job 123\n", "")
        with (
            patch.object(prepare_datasets.subprocess, "run", return_value=result) as submit,
            patch.object(prepare_datasets, "prepare_datasets") as prepare,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(prepare_datasets.main([]), 0)
        prepare.assert_not_called()
        self.assertFalse((self.data / "archives").exists())
        options = submit.call_args.args[0]
        self.assertEqual(options[0], "sbatch")
        for option in ("--account=PAS3560", "--clusters=pitzer", "--partition=cpu-exp", "--cpus-per-task=1", "--time=04:00:00"):
            self.assertIn(option, options)
        self.assertFalse(any(option.startswith("--mem") for option in options))
        self.assertFalse(any("gpu" in option for option in options))
        command = shlex.split(submit.call_args.kwargs["input"].splitlines()[-1])
        self.assertEqual(command[:4], ["srun", "--cpus-per-task=1", sys.executable, "-u"])
        self.assertIn("--run", command)
        self.assertEqual(command[command.index("--data-root") + 1], str(self.data))

    def test_generated_job_runs_pipeline_with_resource_overrides_and_reports_partial_success(self):
        good = self.make_dataset("v001.example's data")
        (self.data / "v002.broken").mkdir()
        report = self.root / "reports with spaces" / "summary.json"
        result = subprocess.CompletedProcess(["sbatch"], 0, "Submitted batch job 456\n", "")
        with (
            patch.object(prepare_datasets.subprocess, "run", return_value=result) as submit,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = prepare_datasets.main([
                "--time", "08:00:00", "--cpus", "2",
                "--account", "PAS1234", "--cluster", "another-cluster", "--partition", "cpu",
                "--report", str(report),
            ])
        self.assertEqual(status, 0)
        options = submit.call_args.args[0]
        for option in ("--time=08:00:00", "--cpus-per-task=2", "--account=PAS1234", "--clusters=another-cluster", "--partition=cpu"):
            self.assertIn(option, options)
        binaries = self.root / "bin"
        binaries.mkdir()
        srun = binaries / "srun"
        srun.write_text('#!/bin/sh\n[ "$1" = "--cpus-per-task=2" ] || exit 90\nshift\nexec "$@"\n')
        srun.chmod(0o755)
        completed = subprocess.run(
            ["bash"], input=submit.call_args.kwargs["input"],
            env={
                **os.environ,
                "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
                "SLURM_CPUS_PER_TASK": "3",
                "SLURM_TRES_PER_TASK": "cpu=1",
            },
            cwd=self.root, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(report.read_text())
        self.assertEqual([item["version"] for item in summary["prepared"]], [good.name])
        self.assertEqual([item["version"] for item in summary["skipped"]], ["v002.broken"])
        self.assertTrue(Path(summary["prepared"][0]["archive"]).is_file())

    def test_all_failed_versions_produce_report_and_nonzero_status(self):
        (self.data / "v001.broken").mkdir()
        report = self.root / "summary.json"
        with contextlib.redirect_stdout(io.StringIO()):
            status = prepare_datasets.main(["--run", "--report", str(report)])
        self.assertEqual(status, 1)
        summary = json.loads(report.read_text())
        self.assertEqual(summary["prepared"], [])
        self.assertEqual(summary["skipped"][0]["version"], "v001.broken")

    def test_empty_dataset_root_exits_with_actionable_error(self):
        with (
            contextlib.redirect_stderr(io.StringIO()) as errors,
            self.assertRaises(SystemExit) as error,
        ):
            prepare_datasets.main(["--run"])
        self.assertEqual(error.exception.code, 1)
        self.assertIn("No dataset versions found", errors.getvalue())

    def test_scheduler_rejection_is_reported_without_running_preparation(self):
        result = subprocess.CompletedProcess(["sbatch"], 1, "", "invalid account")
        with (
            patch.object(prepare_datasets.subprocess, "run", return_value=result),
            patch.object(prepare_datasets, "prepare_datasets") as prepare,
            contextlib.redirect_stderr(io.StringIO()) as errors,
            self.assertRaises(SystemExit) as error,
        ):
            prepare_datasets.main([])
        self.assertEqual(error.exception.code, 1)
        self.assertIn("sbatch failed", errors.getvalue())
        self.assertIn("invalid account", errors.getvalue())
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
