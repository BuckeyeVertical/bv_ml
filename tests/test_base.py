import contextlib
import io
import json
import os
import runpy
import shlex
import subprocess
import sys
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from unittest.mock import Mock, patch

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
            "def train(run):\n"
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
        self.assertEqual(first.wandb_entity, "BuckeyeVertical")
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

    def test_queued_job_reuses_reserved_folder_and_run_id(self):
        submitted = base._reserve_run(self.script, {}, 1)
        started = base._reserve_run(
            self.script, {}, 1,
            run_timestamp=submitted.run_timestamp,
            season=submitted.season,
        )
        self.assertEqual(started, submitted)
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
        self.assertEqual(config.out_dir.parent, self.root / "out/2027-2028/example")


class FrameworkAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root / "models/example.py"
        self.script.parent.mkdir()
        self.adapter = self.script.with_name("__wandb__.py")
        self.run = base.Run(
            data_dir=self.root / "data/v001", out_dir=self.root / "out",
            model_cache_dir=self.root / "cache", slurm={}, model_path=Path("example"),
            run_id="0123456789abcdef0123456789abcdef",
            run_timestamp="20261020T142813000000Z", data_version=1,
        )
        self.run.out_dir.mkdir()
        self.events = self.run.out_dir / "events.txt"
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def model(self, *, fails=False):
        self.script.write_text(
            "from base import slurmable\n"
            "@slurmable\n"
            "def train(run):\n"
            "    with (run.out_dir / 'events.txt').open('a') as output:\n"
            "        output.write('train\\n')\n"
            + ("    raise ValueError('training failed')\n" if fails else "    return run.data_dir\n")
        )
        return runpy.run_path(str(self.script))["train"]

    def write_adapter(self):
        self.adapter.write_text(
            "from contextlib import contextmanager\n"
            "def record(run, message):\n"
            "    with (run.out_dir / 'events.txt').open('a') as output:\n"
            "        output.write(message + '\\n')\n"
            "@contextmanager\n"
            "def track(run):\n"
            "    record(run, 'start:' + str(run.data_dir))\n"
            "    try:\n"
            "        yield\n"
            "    except BaseException:\n"
            "        record(run, 'error')\n"
            "        raise\n"
            "    finally:\n"
            "        record(run, 'end')\n"
        )

    def test_import_and_submission_do_not_load_the_adapter(self):
        self.adapter.write_text("raise AssertionError('Adapter must only load during training')\n")
        with (
            patch.object(base, "_reserve_run", return_value=self.run) as reserve,
            patch.object(base, "_submit") as submit,
            patch.dict(sys.modules, {"wandb": None, "ultralytics": None, "lightly_train": None}),
        ):
            self.model()
            reserve.assert_not_called()
            submit.assert_not_called()
            with patch.object(base.sys, "argv", [str(self.script)]):
                runpy.run_path(str(self.script), run_name="__main__")
        submit.assert_called_once_with(self.script, self.run)
        self.assertFalse(self.events.exists())

    def test_decorated_calls_enter_and_close_the_adapter_on_success_and_failure(self):
        self.write_adapter()
        for fails in (False, True):
            with self.subTest(training_fails=fails):
                self.events.unlink(missing_ok=True)
                train = self.model(fails=fails)
                with patch.object(base, "_reserve_run", return_value=self.run):
                    if fails:
                        with self.assertRaisesRegex(ValueError, "training failed"):
                            train()
                    else:
                        self.assertEqual(train(), self.run.data_dir)
                self.assertEqual(self.events.read_text().splitlines(), [
                    f"start:{self.run.data_dir}", "train", *(["error"] if fails else []), "end",
                ])

    def test_adapter_receives_the_staged_dataset_path(self):
        self.write_adapter()
        train = self.model()
        staged = self.root / "scratch/v001"
        staged.mkdir(parents=True)
        with (
            patch.object(base, "_reserve_run", return_value=self.run),
            patch.dict(os.environ, {"BV_ML_DATA_DIR": str(staged)}),
        ):
            self.assertEqual(train(), staged)
        self.assertEqual(self.events.read_text().splitlines(), [f"start:{staged}", "train", "end"])
        self.assertEqual(self.run.data_dir, self.root / "data/v001")

    def test_only_the_sibling_adapter_is_loaded(self):
        (self.root / "__wandb__.py").write_text("raise AssertionError('Wrong adapter')\n")
        train = self.model()
        with patch.object(base, "_reserve_run", return_value=self.run), patch.dict(sys.modules, {"wandb": None}):
            self.assertEqual(train(), self.run.data_dir)
        self.assertEqual(self.events.read_text(), "train\n")

    def test_broken_adapter_is_reported_before_training(self):
        self.adapter.write_text("raise RuntimeError('tracking setup failed')\n")
        train = self.model()
        with patch.object(base, "_reserve_run", return_value=self.run), self.assertRaisesRegex(RuntimeError, "tracking setup failed"):
            train()
        self.assertFalse(self.events.exists())


class ExecutionTests(unittest.TestCase):
    def test_training_return_values_and_errors_pass_through_without_wandb(self):
        run = object()
        for fails in (False, True):
            with self.subTest(training_fails=fails):
                callback = Mock(side_effect=ValueError("training failed")) if fails else Mock(return_value="training result")

                def train(received, callback=callback):
                    return callback(received)

                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch.dict(sys.modules, {"wandb": None}),
                ):
                    if fails:
                        with self.assertRaisesRegex(ValueError, "training failed"):
                            base._execute(train, run)
                    else:
                        self.assertEqual(base._execute(train, run), "training result")
                callback.assert_called_once_with(run)

    def test_every_worker_executes_training_without_wrapper_tracking(self):
        for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
            with self.subTest(rank_variable=key):
                run = object()
                callback = Mock(return_value="worker result")

                def train(received, callback=callback):
                    return callback(received)

                with (
                    patch.dict(os.environ, {key: "1"}, clear=True),
                    patch.dict(sys.modules, {"wandb": None}),
                ):
                    self.assertEqual(base._execute(train, run), "worker result")
                callback.assert_called_once_with(run)


if __name__ == "__main__":
    unittest.main()
