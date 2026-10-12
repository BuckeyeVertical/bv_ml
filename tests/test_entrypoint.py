import runpy
import shlex
import subprocess
import sys
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from base import ROOT, SlurmJob
from models.ultralytics.framework import UltralyticsModel


def reserve_run(cls, run=None):
    cls.config = replace(cls.config, _run=9 if run is None else run)


class EntrypointTests(unittest.TestCase):
    def test_model_help_from_outside_repository(self):
        with TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "models/ultralytics/yolo26s.py"), "--help"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: yolo26s.py", result.stdout)

    def test_submission_uses_model_script_and_same_python(self):
        arguments = [str(ROOT / "models/ultralytics/yolo26s.py")]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(SlurmJob, "prepare_run", classmethod(reserve_run)),
            patch("base.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="submitted", stderr="")) as submit,
        ):
            runpy.run_path(arguments[0], run_name="__main__")
        kwargs = submit.call_args.kwargs
        command = shlex.split(kwargs["input"].strip().splitlines()[-1])
        self.assertEqual(command, [
            "srun", sys.executable,
            str(ROOT / "models/ultralytics/yolo26s.py"), "--train", "--run", "9",
        ])
        self.assertEqual(submit.call_args.args[0], ["sbatch", f"--chdir={ROOT}", "--clusters=pitzer"])

    def test_training_forwards_reserved_run_and_skips_submission(self):
        arguments = [str(ROOT / "models/ultralytics/yolo26s.py"), "--train", "--run", "12"]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(SlurmJob, "prepare_run", classmethod(reserve_run)),
            patch.object(UltralyticsModel, "train", autospec=True) as train,
            patch.object(SlurmJob, "submit") as submit,
        ):
            runpy.run_path(arguments[0], run_name="__main__")
        submit.assert_not_called()
        train.assert_called_once()
        self.assertEqual(train.call_args.args[0].config._run, 12)
        self.assertEqual(train.call_args.args[0].config.job_name, "yolo26s")


if __name__ == "__main__":
    unittest.main()
