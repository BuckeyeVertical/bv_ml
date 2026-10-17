import contextlib
import io
import json
import runpy
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import base


class DataVersionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for version in (1, 2):
            (self.root / "data" / f"v{version:03d}.example").mkdir(parents=True)
        self.script = self.root / "models" / "example.py"
        self.script.parent.mkdir()
        root_patch = patch.object(base, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write_model(self, decorator, module_version=99):
        self.script.write_text(
            "from base import slurmable\n"
            f"DATA_VERSION = {module_version}\n"
            f"{decorator}\n"
            "def train(config):\n"
            "    raise AssertionError('Submission must not execute training')\n"
        )

    def submit(self):
        result = subprocess.CompletedProcess(["sbatch"], 0, "Submitted batch job 123\n", "")
        with (
            patch.object(base.subprocess, "run", return_value=result) as submit,
            patch.object(base.sys, "argv", [str(self.script)]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            runpy.run_path(str(self.script), run_name="__main__")
        submit.assert_called_once()
        return submit.call_args.kwargs["input"]

    def assert_submitted_version(self, script, version):
        command = shlex.split(script.splitlines()[-1])
        self.assertEqual(command[command.index("--data-version") + 1], str(version))
        self.assertIn(f".dv{version:03d}.", script)
        directives = [line for line in script.splitlines() if line.startswith("#SBATCH")]
        self.assertFalse(any("DATA" in line or "data-version" in line for line in directives))

    def test_bare_and_empty_decorators_default_to_latest_without_module_lookup(self):
        for decorator in ("@slurmable", "@slurmable()"):
            with self.subTest(decorator=decorator):
                self.write_model(decorator)
                self.assert_submitted_version(self.submit(), 2)

    def test_explicit_version_is_reserved_and_slurm_options_are_preserved(self):
        self.write_model('@slurmable(DATA_VERSION=1, cpus_per_task=12, time="02:00:00")')
        script = self.submit()
        self.assert_submitted_version(script, 1)
        self.assertIn("#SBATCH --cpus-per-task=12", script)
        self.assertIn("#SBATCH --time=02:00:00", script)

    def test_latest_is_resolved_and_pinned_before_submission(self):
        for decorator in ("@slurmable", "@slurmable(DATA_VERSION=-1)"):
            with self.subTest(decorator=decorator):
                self.write_model(decorator)
                script = self.submit()
                self.assert_submitted_version(script, 2)
                command = shlex.split(script.splitlines()[-1])
                newer = self.root / "data" / "v3.new"
                newer.mkdir()
                with (
                    patch.object(base.sys, "argv", command[2:]),
                    patch.object(base, "_execute") as execute,
                    patch.object(base.subprocess, "run") as submit,
                ):
                    runpy.run_path(str(self.script), run_name="__main__")
                submit.assert_not_called()
                config = execute.call_args.args[1]
                self.assertEqual(config.data_version, 2)
                self.assertIn(str(config.out_dir / "logs" / "slurm-%j.out"), script)
                newer.rmdir()

    def test_direct_execution_uses_selected_dataset(self):
        self.write_model("@slurmable()")
        with (
            patch.object(base.sys, "argv", [str(self.script), "--train"]),
            patch.object(base, "_execute") as execute,
            patch.object(base.subprocess, "run") as submit,
        ):
            runpy.run_path(str(self.script), run_name="__main__")
        submit.assert_not_called()
        config = execute.call_args.args[1]
        self.assertEqual(config.data_version, 2)
        self.assertEqual(config.data_dir, self.root / "data" / "v002.example")

    def test_padded_names_preserve_counters_and_distinguish_same_day_runs(self):
        (self.root / "run_counts.json").write_text(json.dumps({"example": 16}))
        first = base._reserve_run(self.script, {}, 1, run_timestamp="20261017T160204423869Z")
        second = base._reserve_run(self.script, {}, 1, run_timestamp="20261017T160205423869Z")
        self.assertEqual(first.out_dir.name, "v017.dv001.10-17-26")
        self.assertEqual(second.out_dir.name, "v018.dv001.10-17-26")
        self.assertEqual(first.run_timestamp, "20261017T160204423869Z")
        self.assertEqual(json.loads((self.root / "run_counts.json").read_text())["example"], 18)
        self.assertTrue((first.out_dir / "logs").is_dir())

    def test_legacy_dataset_names_and_versions_above_999(self):
        legacy = self.root / "data" / "v10.legacy"
        legacy.mkdir()
        config = base._reserve_run(self.script, {}, -1)
        self.assertEqual(config.data_dir, legacy)
        self.assertTrue(config.out_dir.name.startswith("v001.dv010."))
        (self.root / "data" / "v1000.large").mkdir()
        config = base._reserve_run(self.script, {}, -1, run_number=1000)
        self.assertTrue(config.out_dir.name.startswith("v1000.dv1000."))

    def test_jobs_submitted_before_naming_change_reuse_existing_folder(self):
        legacy = self.root / "out" / "example" / "v14.dv1.10-17-16:02-26"
        (legacy / "logs").mkdir(parents=True)
        config = base._reserve_run(
            self.script, {}, 1, run_number=14, run_timestamp="20261017T160204423869Z"
        )
        self.assertEqual(config.out_dir, legacy)
        self.assertFalse((legacy.parent / "v014.dv001.10-17-26").exists())


if __name__ == "__main__":
    unittest.main()
