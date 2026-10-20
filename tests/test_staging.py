import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from PIL import Image

import base
import prepare_datasets


class DatasetStagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="staging tests '")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = patch.object(base, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.version = self.root / "data" / "v001.example's data"
        images = self.version / "yolo/images/train"
        images.mkdir(parents=True)
        self.image = images / "a.png"
        Image.new("RGB", (20, 10), "white").save(self.image)
        (self.version / "yolo/data.yaml").write_text("train: images/train\nnames: [person]\n")
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_datasets.convert_yolo_to_coco(self.version)
            self.archive = prepare_datasets.archive_dataset(self.version)
        self.script = self.root / "models/example.py"
        self.script.parent.mkdir()
        self.script.write_text("# Training is replaced by the test srun executable.\n")
        self.config = base._reserve_run(self.script, {"nodes": 1, "cpus-per-task": 3}, 1)
        self.scratch = self.root / "node scratch"
        self.scratch.mkdir()
        self.capture = self.root / "training.json"
        binaries = self.root / "bin"
        binaries.mkdir()
        srun = binaries / "srun"
        srun.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1] != '--cpus-per-task=3': sys.exit(90)\n"
            "data = Path(os.environ['BV_ML_DATA_DIR'])\n"
            "link = data / 'coco/images/train'\n"
            "result = {\n"
            "    'data_dir': str(data),\n"
            "    'image': (link / 'a.png').read_bytes().hex(),\n"
            "    'link': str(link.resolve()),\n"
            "    'symlink': link.is_symlink(),\n"
            "    'args': sys.argv[1:],\n"
            "}\n"
            "Path(os.environ['STAGING_CAPTURE']).write_text(json.dumps(result))\n"
            "sys.exit(int(os.environ.get('TRAIN_EXIT_CODE', '0')))\n"
        )
        srun.chmod(0o755)
        self.env = {
            **os.environ,
            "TMPDIR": str(self.scratch),
            "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
            "STAGING_CAPTURE": str(self.capture),
            "SLURM_CPUS_PER_TASK": "3",
            "SLURM_TRES_PER_TASK": "cpu=1",
        }
        self.env.pop("BV_ML_DATA_DIR", None)

    def batch_script(self):
        result = subprocess.CompletedProcess(["sbatch"], 0, "Submitted batch job 123\n", "")
        with (
            patch.object(base.subprocess, "run", return_value=result) as submit,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            base._submit(self.script, self.config)
        return submit.call_args.kwargs["input"]

    def test_batch_stages_data_before_training_and_cleans_only_its_copy(self):
        batch = self.batch_script()
        neighbor = self.scratch / "other-job.txt"
        neighbor.write_text("preserve me")
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code):
                result = subprocess.run(
                    ["bash"], input=batch, cwd=self.root,
                    env={**self.env, "TRAIN_EXIT_CODE": str(exit_code)},
                    text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, exit_code, result.stderr)
                captured = json.loads(self.capture.read_text())
                staged = Path(captured["data_dir"])
                self.assertTrue(staged.is_relative_to(self.scratch))
                self.assertEqual(staged.name, self.version.name)
                self.assertEqual(captured["image"], self.image.read_bytes().hex())
                self.assertTrue(captured["symlink"])
                self.assertEqual(Path(captured["link"]), staged / "yolo/images/train")
                self.assertIn(self.config.run_timestamp, captured["args"])
                self.assertFalse(staged.parent.exists())
                self.assertEqual(neighbor.read_text(), "preserve me")
                self.assertTrue(self.archive.is_file())
                self.assertTrue(self.image.is_file())

    def test_failed_extraction_never_starts_training_and_cleans_partial_data(self):
        self.archive.write_bytes(b"invalid archive")
        result = subprocess.run(
            ["bash"], input=self.batch_script(), env=self.env,
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertTrue(self.image.is_file())

    def test_missing_tmpdir_stops_before_training(self):
        env = self.env.copy()
        env.pop("TMPDIR")
        result = subprocess.run(
            ["bash"], input=self.batch_script(), env=env,
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TMPDIR is required", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_missing_archive_and_multiple_nodes_are_rejected_before_submission(self):
        self.archive.unlink()
        with patch.object(base.subprocess, "run") as submit:
            with self.assertRaisesRegex(FileNotFoundError, "prepare_datasets.py"):
                base._submit(self.script, self.config)
            with self.assertRaisesRegex(ValueError, "single-node"):
                base._submit(self.script, replace(self.config, slurm={"nodes": 2}))
        submit.assert_not_called()

    def test_wrapper_changes_only_the_dataset_path(self):
        staged = self.scratch / self.version.name
        shutil.copytree(self.version, staged, symlinks=True)
        wandb = MagicMock()
        with (
            patch.dict(os.environ, {"BV_ML_DATA_DIR": str(staged)}),
            patch.dict(sys.modules, {"wandb": wandb}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            received = base._execute(lambda config: config, self.config)
        self.assertEqual(received, replace(self.config, data_dir=staged))
        self.assertEqual(received.train_out_dir, self.config.train_out_dir)
        self.assertEqual(received.wandb, self.config.wandb)
        self.assertEqual(self.config.data_dir, self.version)
        self.assertEqual((received.data_dir / "coco/images/train/a.png").read_bytes(), self.image.read_bytes())
        self.assertEqual(wandb.init.call_args.kwargs["dir"], str(self.config.out_dir))

    def test_invalid_staged_paths_stop_before_logging_or_training(self):
        wrong_version = self.scratch / "v002.other"
        wrong_version.mkdir()
        file_path = self.scratch / self.version.name
        file_path.write_text("not a directory")
        cases = [
            (self.scratch / "missing" / self.version.name, FileNotFoundError),
            (wrong_version, ValueError),
            (file_path, ValueError),
        ]
        for staged, error in cases:
            with self.subTest(staged=staged):
                wandb = MagicMock()
                train = Mock()
                with (
                    patch.dict(os.environ, {"BV_ML_DATA_DIR": str(staged)}),
                    patch.dict(sys.modules, {"wandb": wandb}),
                    self.assertRaises(error),
                ):
                    base._execute(train, self.config)
                wandb.init.assert_not_called()
                train.assert_not_called()

    def test_archive_reuses_and_explicitly_rebuilds_completed_archives(self):
        original = self.archive.read_bytes()
        modified = self.archive.stat().st_mtime_ns
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_datasets.archive_dataset(self.version)
        self.assertEqual(self.archive.read_bytes(), original)
        self.assertEqual(self.archive.stat().st_mtime_ns, modified)
        (self.version / "coco/extra.json").write_text("{}")
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_datasets.archive_dataset(self.version, force=True)
        with tarfile.open(self.archive) as archive:
            self.assertEqual(archive.extractfile(f"{self.version.name}/coco/extra.json").read(), b"{}")
            self.assertTrue(archive.getmember(f"{self.version.name}/coco/images/train").issym())

    def test_concurrent_archive_commands_publish_one_complete_archive(self):
        self.archive.unlink()
        with (
            patch.object(prepare_datasets.tarfile, "open", wraps=tarfile.open) as archive_open,
            contextlib.redirect_stdout(io.StringIO()),
            ThreadPoolExecutor(max_workers=3) as executor,
        ):
            results = list(executor.map(prepare_datasets.archive_dataset, [self.version] * 3))
        self.assertEqual(results, [self.archive] * 3)
        self.assertEqual(archive_open.call_count, 1)
        with tarfile.open(self.archive) as archive:
            self.assertEqual(archive.extractfile(f"{self.version.name}/yolo/images/train/a.png").read(), self.image.read_bytes())

    def test_failed_rebuild_keeps_previous_archive_and_removes_temporary_output(self):
        original = self.archive.read_bytes()
        with (
            patch.object(prepare_datasets.tarfile.TarFile, "add", side_effect=OSError("archive write failed")),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(OSError, "archive write failed"),
        ):
            prepare_datasets.archive_dataset(self.version, force=True)
        self.assertEqual(self.archive.read_bytes(), original)
        self.assertEqual(list(self.archive.parent.glob(f".{self.version.name}-*")), [])


if __name__ == "__main__":
    unittest.main()
