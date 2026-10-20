import contextlib
import io
import json
import runpy
import shlex
import subprocess
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from unittest.mock import patch

import base


class DataVersionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for version in (1, 2):
            version_dir = self.root / "data" / f"v{version:03d}.example"
            version_dir.mkdir(parents=True)
            archives = self.root / "data" / "archives"
            archives.mkdir(exist_ok=True)
            with tarfile.open(archives / f"{version_dir.name}.tar", "w") as archive:
                archive.add(version_dir, arcname=version_dir.name)
        self.script = self.root / "models" / "example.py"
        self.script.parent.mkdir()
        root_patch = patch.object(base, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        clock_patch = patch.object(base, "datetime", wraps=datetime)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)
        ticks = count()
        self.clock.now.side_effect = lambda tz: datetime(2026, 10, 20, 18, 28, 13, 123456, tzinfo=tz) + timedelta(seconds=next(ticks))

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
        self.assertIn(f".dv{version:03d}/logs/", script)
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
        self.assertIn("--cpus-per-task=12", shlex.split(script.splitlines()[-1]))
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
                    patch.object(base.sys, "argv", command[command.index(base.sys.executable) + 1:]),
                    patch.object(base, "_execute") as execute,
                    patch.object(base.subprocess, "run") as submit,
                    patch.object(base, "SEASON", "2027-2028"),
                ):
                    runpy.run_path(str(self.script), run_name="__main__")
                submit.assert_not_called()
                config = execute.call_args.args[1]
                self.assertEqual(config.data_version, 2)
                self.assertEqual(config.season, "2026-2027")
                self.assertEqual(config.wandb.project, "season-2026-2027")
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

    def test_timestamp_names_and_per_run_metadata(self):
        first = base._reserve_run(self.script, {}, 1)
        second = base._reserve_run(self.script, {}, 1)
        self.assertEqual(first.out_dir.name, "example.26-10-20-182813.dv001")
        self.assertEqual(second.out_dir.name, "example.26-10-20-182814.dv001")
        self.assertEqual(first.out_dir.parent, self.root / "out/2026-2027/example")
        self.assertEqual(first.run_timestamp, "20261020T182813123456Z")
        self.assertRegex(first.run_id, r"^[0-9a-f]{32}$")
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(json.loads((first.out_dir / "run.json").read_text()), {
            "run_id": first.run_id,
            "season": "2026-2027",
            "model_path": "example",
            "run_timestamp": first.run_timestamp,
            "data_version": 1,
        })
        self.assertTrue((first.out_dir / "logs").is_dir())

    def test_dataset_versions_sort_numerically_and_expand_above_999(self):
        dataset = self.root / "data" / "v010.example"
        dataset.mkdir()
        config = base._reserve_run(self.script, {}, -1)
        self.assertEqual(config.data_dir, dataset)
        self.assertTrue(config.out_dir.name.endswith(".dv010"))
        (self.root / "data" / "v1000.large").mkdir()
        config = base._reserve_run(self.script, {}, -1)
        self.assertTrue(config.out_dir.name.endswith(".dv1000"))

    def test_queued_job_reuses_reserved_folder_and_wandb_id(self):
        submitted = base._reserve_run(self.script, {}, 1)
        started = base._reserve_run(
            self.script, {}, 1,
            run_timestamp=submitted.run_timestamp,
            season=submitted.season,
        )
        self.assertEqual(started, submitted)
        self.assertEqual(started.wandb, submitted.wandb)
        self.clock.now.assert_called_once_with(timezone.utc)

    def test_same_second_collision_preserves_existing_run(self):
        self.clock.now.side_effect = None
        self.clock.now.return_value = datetime(2026, 10, 20, 18, 28, 13, tzinfo=timezone.utc)
        first = base._reserve_run(self.script, {}, 1)
        original = (first.out_dir / "run.json").read_bytes()
        marker = first.out_dir / "logs/slurm-123.out"
        marker.write_text("active run")
        with self.assertRaisesRegex(FileExistsError, "Run output already exists"):
            base._reserve_run(self.script, {}, 1)
        self.assertEqual((first.out_dir / "run.json").read_bytes(), original)
        self.assertEqual(marker.read_text(), "active run")

    def test_queued_job_rejects_mismatched_metadata(self):
        submitted = base._reserve_run(self.script, {}, 1)
        path = submitted.out_dir / "run.json"
        metadata = json.loads(path.read_text())
        metadata["data_version"] = 2
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "metadata does not match"):
            base._reserve_run(self.script, {}, 1, run_timestamp=submitted.run_timestamp)

    def test_season_validation_and_separate_season_folders(self):
        for season in ("2026/2027", "2026-2028", "../outside"):
            with self.subTest(season=season), self.assertRaisesRegex(ValueError, "Season"):
                base._reserve_run(self.script, {}, 1, season=season)
        config = base._reserve_run(self.script, {}, 1, season="2027-2028")
        self.assertEqual(config.season, "2027-2028")
        self.assertEqual(config.wandb.project, "season-2027-2028")
        self.assertEqual(config.out_dir.parent, self.root / "out/2027-2028/example")


class WandbTests(unittest.TestCase):
    def test_run_context_logs_and_closes_on_success_and_failure_for_any_framework(self):
        for model_path, fails in (("ultralytics/yolo26s", False), ("custom/example", True)):
            with self.subTest(model_path=model_path):
                self.check_run_context(model_path, fails)

    def check_run_context(self, model_path, fails):
        import wandb

        self.addCleanup(wandb.teardown)
        real_init = wandb.init
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = base.Run(
                data_dir=root / "data",
                out_dir=root,
                model_cache_dir=root / "cache",
                slurm={},
                model_path=Path(model_path),
                run_id="0123456789abcdef0123456789abcdef",
                run_timestamp="20261020T182813123456Z",
                data_version=1,
            )
            runs = []

            def offline_init(**kwargs):
                self.assertEqual(kwargs["mode"], "online")
                self.assertEqual(kwargs["entity"], "BuckeyeVertical")
                self.assertEqual(kwargs["project"], "season-2026-2027")
                self.assertEqual(kwargs["id"], config.run_id)
                self.assertEqual(kwargs["group"], model_path)
                self.assertEqual(kwargs["name"], config.out_dir.name)
                self.assertEqual(kwargs["dir"], str(config.out_dir))
                self.assertEqual(config.wandb.dir, kwargs["dir"])
                self.assertEqual(config.wandb.mode, kwargs["mode"])
                kwargs["mode"] = "offline"
                kwargs["settings"] = wandb.Settings(console="off", quiet=True, disable_git=True, finish_timeout=10)
                return real_init(**kwargs)

            def train(received):
                self.assertIs(received, config)
                run = wandb.run
                runs.append(run)
                self.assertEqual(received.wandb.entity, run.entity)
                self.assertEqual(received.wandb.project, run.project)
                self.assertEqual(received.wandb.group, run.group)
                self.assertEqual(received.wandb.name, run.name)
                self.assertEqual(received.wandb.id, run.id)
                self.assertEqual(run.config["run_id"], config.run_id)
                self.assertEqual(run.config["model_path"], model_path)
                wandb.log({"loss": 1.25})
                self.assertEqual(dict(run.summary)["loss"], 1.25)
                if fails:
                    raise ValueError("training failed")
                return "training result"

            with (
                patch("wandb.init", side_effect=offline_init),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                if fails:
                    with self.assertRaisesRegex(ValueError, "training failed"):
                        base._execute(train, config)
                else:
                    self.assertEqual(base._execute(train, config), "training result")
            self.assertIsNone(wandb.run)
            self.assertEqual(len(runs), 1)
            self.assertTrue(runs[0]._is_finished)


if __name__ == "__main__":
    unittest.main()
