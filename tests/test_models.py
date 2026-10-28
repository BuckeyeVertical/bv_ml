import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import base


class LightlyLaunchTests(unittest.TestCase):
    def test_one_slurm_task_launches_two_workers_with_one_wandb_run(self):
        source = (base.ROOT / "models/lightly/ltdetrv2-s.py").read_text()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data/v002/yolo").mkdir(parents=True)
            (root / "data/v002/yolo/data.yaml").write_text(
                "train: images/train\nval: images/val\nnames: [person]\n"
            )
            script = root / "models/lightly/ltdetrv2-s.py"
            script.parent.mkdir(parents=True)
            for adapter in ("__data__.py", "__wandb__.py"):
                (script.parent / adapter).write_text(
                    (base.ROOT / "models/lightly" / adapter).read_text()
                )
            # Exercise the real model and Lightly's argument validation, then
            # replace training with a two-process CPU collective. No downloads,
            # GPU allocation, dataset scan, or online W&B connection is needed.
            script.write_text(textwrap.dedent('''\
                import json
                import os
                from pathlib import Path

                import torch
                import wandb
                from lightning_fabric import Fabric
                from lightly_train._commands import train_task, train_task_helpers

                import base

                base.ROOT = Path(__file__).resolve().parents[2]
                original_slurmable = base.slurmable

                def two_worker_slurmable(**options):
                    options["gpus_per_node"] = 2
                    return original_slurmable(**options)

                base.slurmable = two_worker_slurmable

                real_init = wandb.init

                def offline_init(**kwargs):
                    marker = base.ROOT / f"wandb-init-{os.getpid()}.json"
                    marker.write_text(json.dumps({"id": kwargs["id"]}))
                    kwargs["mode"] = "offline"
                    kwargs["settings"] = wandb.Settings(console="off", quiet=True, disable_git=True, finish_timeout=10)
                    return real_init(**kwargs)

                wandb.init = offline_init

                def cpu_train(config):
                    fabric = Fabric(
                        accelerator="cpu",
                        devices=config.devices,
                        strategy=config.strategy,
                        precision="32-true",
                    )
                    fabric.launch()
                    total = fabric.all_reduce(torch.tensor(1.0), reduce_op="sum")
                    out = Path(config.out)
                    out.mkdir(parents=True, exist_ok=True)
                    args = train_task_helpers.get_logger_args(
                        steps=config.steps, val_steps=2, logger_args=config.logger_args,
                    )
                    args.tensorboard = None
                    fabric.loggers.extend(train_task_helpers.get_loggers(
                        args, out=out, resume_interrupted=False,
                    ))
                    fabric.log_dict({"worker_sum": total.item()}, step=1)
                    (out / f"worker-{fabric.global_rank}.json").write_text(json.dumps({
                        "rank": fabric.global_rank,
                        "world_size": fabric.world_size,
                        "sum": total.item(),
                        "slurm_tasks": os.environ["SLURM_NTASKS"],
                        "slurm_job_name": os.environ["SLURM_JOB_NAME"],
                    }))
                    fabric.barrier()
                    torch.distributed.destroy_process_group()

                train_task._train_task_from_config = cpu_train

            ''') + source)
            with patch.object(base, "ROOT", root):
                config = base._reserve_run(script, {"ntasks": 1, "gpus-per-node": 2}, 2)
            env = os.environ.copy()
            for key in (
                "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "NODE_RANK", "GROUP_RANK", "BV_ML_DATA_DIR",
            ):
                env.pop(key, None)
            env.update({
                "PYTHONPATH": str(base.ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1",
                "SLURM_NTASKS": "1",
                "SLURM_NTASKS_PER_NODE": "1",
                "SLURM_NNODES": "1",
                "SLURM_PROCID": "0",
                "SLURM_LOCALID": "0",
                "SLURM_NODEID": "0",
                "SLURM_JOB_ID": "123",
                "SLURM_JOB_NAME": "ltdetrv2-s",
            })
            result = subprocess.run(
                [
                    sys.executable, "-B", str(script), "--train",
                    "--timestamp", config.run_timestamp,
                    "--season", config.season, "--data-version", "2",
                ],
                cwd=root, env=env, text=True, capture_output=True, timeout=180, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for rank in (0, 1):
                worker = json.loads((config.train_out_dir / f"worker-{rank}.json").read_text())
                self.assertEqual(worker, {
                    "rank": rank, "world_size": 2, "sum": 2.0,
                    "slurm_tasks": "1", "slurm_job_name": "ltdetrv2-s",
                })
            tracking = list(root.glob("wandb-init-*.json"))
            self.assertEqual(len(tracking), 1, "Distributed workers opened duplicate W&B runs")
            self.assertEqual(json.loads(tracking[0].read_text())["id"], config.run_id)


class LightlyWandbTests(unittest.TestCase):
    def test_native_logger_starts_one_run_and_logs_metrics_and_images(self):
        import lightly_train
        import wandb
        from lightly_train._commands import train_task, train_task_helpers
        from PIL import Image

        self.addCleanup(wandb.teardown)
        real_init = wandb.init
        original_train = lightly_train.train_object_detection
        original_environment = {key: os.environ.get(key) for key in ("WANDB_ENTITY", "WANDB_RUN_GROUP")}
        model = runpy.run_path(str(base.ROOT / "models/lightly/ltdetrv2-s.py"))
        for fails in (False, True):
            with self.subTest(training_fails=fails), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_root = root / "data/v002/yolo"
                data_root.mkdir(parents=True)
                (data_root / "data.yaml").write_text(
                    "train: images/train\nval: images/val\ntest: images/test\n"
                    "nc: 2\nnames: {0: Coverall, 1: Mask}\n"
                )
                out = root / "run"
                out.mkdir()
                config = base.Run(
                    data_dir=data_root.parent, out_dir=out,
                    model_cache_dir=root / "cache", slurm={"gpus-per-node": 1},
                    model_path=Path("lightly/ltdetrv2-s"),
                    run_id="0123456789abcdef0123456789abcdef",
                    run_timestamp="20261020T142813000000Z", data_version=2,
                    wandb_entity="configured-team",
                )
                runs = []
                metric_definitions = []

                def offline_init(runs=runs, metric_definitions=metric_definitions, **kwargs):
                    kwargs["mode"] = "offline"
                    kwargs["settings"] = wandb.Settings(console="off", quiet=True, disable_git=True, finish_timeout=10)
                    run = real_init(**kwargs)
                    define_metric = run.define_metric

                    def record_metric(*args, **options):
                        metric_definitions.append((args, options))
                        return define_metric(*args, **options)

                    run.define_metric = record_metric
                    runs.append(run)
                    return run

                def log_without_training(config, data_root=data_root, runs=runs, fails=fails):
                    # Exercise real Lightly argument validation and logger creation;
                    # only model training/downloads and the W&B network are replaced.
                    self.assertEqual(config.data.names, {0: "Coverall", 1: "Mask"})
                    self.assertEqual(Path(config.data.path), data_root)
                    self.assertEqual(config.data.train, Path("images/train"))
                    self.assertEqual(config.data.val, Path("images/val"))
                    args = train_task_helpers.get_logger_args(steps=config.steps, val_steps=2, logger_args=config.logger_args)
                    self.assertIsNone(wandb.run)
                    self.assertEqual(args.wandb.version, "0123456789abcdef0123456789abcdef")
                    args.tensorboard = None
                    logger, = train_task_helpers.get_loggers(args, out=config.out, resume_interrupted=False)
                    self.assertIs(logger.experiment, runs[0])
                    logger.log_hyperparams({"steps": config.steps, "batch_size": config.batch_size})
                    self.assertEqual(runs[0].config["steps"], config.steps)
                    self.assertEqual(runs[0].config["batch_size"], config.batch_size)
                    logger.log_metrics({"train_loss/loss": 1.25, "lr": 0.0001}, step=10)
                    logger.log_metrics({"val_metric/map": 0.65}, step=20)
                    logger.log_image("train_samples", [Image.new("RGB", (4, 4))], step=20)
                    summary = dict(runs[0].summary)
                    self.assertEqual(summary["train_loss/loss"], 1.25)
                    self.assertEqual(summary["val_metric/map"], 0.65)
                    self.assertEqual(summary["trainer/global_step"], 20)
                    self.assertIn("train_samples", summary)
                    if fails:
                        raise ValueError("training failed")

                with (
                    patch.dict(os.environ, {
                        "RANK": "0", "LOCAL_RANK": "0", "SLURM_PROCID": "0",
                        "WANDB_ENTITY": "environment-team",
                    }),
                    patch("wandb.init", side_effect=offline_init) as initialize,
                    patch.object(train_task, "_train_task_from_config", side_effect=log_without_training),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    if fails:
                        with self.assertRaisesRegex(ValueError, "training failed"):
                            base._execute(model["train"].__wrapped__, config)
                    else:
                        base._execute(model["train"].__wrapped__, config)
                    self.assertEqual(os.environ["WANDB_ENTITY"], "environment-team")
                initialize.assert_called_once()
                self.assertIs(lightly_train.train_object_detection, original_train)
                self.assertEqual(
                    {key: os.environ.get(key) for key in original_environment}, original_environment,
                )
                self.assertIsNone(wandb.run)
                self.assertTrue(runs[0]._is_finished)
                self.assertEqual(runs[0].entity, "configured-team")
                self.assertEqual(runs[0].project, "season-2026-2027")
                self.assertEqual(runs[0].group, "lightly/ltdetrv2-s")
                self.assertEqual(runs[0]._exit_code, int(fails))
                self.assertIn((("trainer/global_step",), {}), metric_definitions)
                self.assertIn((("*",), {"step_metric": "trainer/global_step", "step_sync": True}), metric_definitions)


class YoloWandbTests(unittest.TestCase):
    def test_only_the_training_process_on_rank_zero_starts_wandb(self):
        import cloudpickle
        import torch
        import wandb
        from ultralytics import settings
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.engine import trainer as trainer_module
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.utils import callbacks
        from ultralytics.utils.callbacks import wb

        self.addCleanup(wandb.teardown)
        real_init = wandb.init
        model = runpy.run_path(str(base.ROOT / "models/ultralytics/yolo26s.py"))
        cases = [(-1, -1, 2, False), (0, 0, 2, False), (1, 1, 2, False), (-1, -1, 1, True)]
        for rank, local_rank, devices, fails in cases:
            with self.subTest(rank=rank, devices=devices, fails=fails), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                out = root / "run"
                out.mkdir()
                run = base.Run(
                    data_dir=root / "data", out_dir=out,
                    model_cache_dir=root / "cache", slurm={"gpus-per-node": devices},
                    model_path=Path("ultralytics/yolo26s"),
                    run_id="0123456789abcdef0123456789abcdef",
                    run_timestamp="20261020T142813000000Z", data_version=1,
                    wandb_entity="configured-team",
                )
                runs = []
                original_callbacks = callbacks.get_default_callbacks()

                def offline_init(runs=runs, **kwargs):
                    kwargs["mode"] = "offline"
                    kwargs["settings"] = wandb.Settings(console="off", quiet=True, disable_git=True, finish_timeout=10)
                    tracking = real_init(**kwargs)
                    runs.append(tracking)
                    return tracking

                def add_native_wandb(trainer):
                    for event in ("on_pretrain_routine_start", "on_train_epoch_end", "on_fit_epoch_end", "on_train_end"):
                        trainer.callbacks[event].append(getattr(wb, event))

                def train_without_gpus(run=run, devices=devices, fails=fails, **options):
                    self.assertIsNone(wandb.run, "The DDP parent must not open a tracking run")
                    self.assertEqual(options["data"], str(run.data_dir / "yolo/data.yaml"))
                    self.assertEqual(options["device"], list(range(devices)))
                    # DDP workers receive serialized copies of the model callbacks.
                    registered = cloudpickle.loads(cloudpickle.dumps(callbacks.get_default_callbacks()))
                    # Run the real trainer's startup and rank checks. Replace only
                    # device selection, dataset loading, and unrelated integrations.
                    trainer = DetectionTrainer(
                        cfg={**DEFAULT_CFG_DICT, "save_dir": ""},
                        overrides={"model": "yolo26s.pt", **options}, _callbacks=registered,
                    )
                    if wandb.run is not None:
                        trainer.epoch = 1
                        trainer.tloss = 1.25
                        trainer.lr = {"lr/pg0": 0.0001}
                        trainer.label_loss_items = lambda loss, prefix: {f"{prefix}/loss": loss}
                        trainer.validator = SimpleNamespace(plots={})
                        trainer.metrics = {"metrics/map": 0.65}
                        wb.on_train_epoch_end(trainer)
                        wb.on_fit_epoch_end(trainer)
                        self.assertEqual(dict(wandb.run.summary)["train/loss"], 1.25)
                        self.assertEqual(dict(wandb.run.summary)["metrics/map"], 0.65)
                        if fails:
                            raise ValueError("training failed")

                with (
                    patch("ultralytics.YOLO") as yolo,
                    patch.dict(os.environ, {"WANDB_ENTITY": "environment-team"}),
                    patch.object(settings, "update") as update,
                    patch.object(wb, "wb", wandb),
                    patch.object(callbacks, "add_integration_callbacks", side_effect=add_native_wandb),
                    patch.object(trainer_module, "RANK", rank),
                    patch.object(trainer_module, "LOCAL_RANK", local_rank),
                    patch.object(trainer_module, "select_device", return_value=torch.device("cuda", 0)),
                    patch.object(trainer_module, "init_seeds"),
                    patch.object(trainer_module, "torch_distributed_zero_first", side_effect=lambda _: contextlib.nullcontext()),
                    patch.object(DetectionTrainer, "get_dataset", return_value={"names": {0: "person"}, "nc": 1}),
                    patch("wandb.init", side_effect=offline_init) as initialize,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    yolo.return_value.train.side_effect = train_without_gpus
                    if fails:
                        with self.assertRaisesRegex(ValueError, "training failed"):
                            base._execute(model["train"].__wrapped__, run)
                    else:
                        base._execute(model["train"].__wrapped__, run)
                update.assert_called_once_with({"wandb": True})
                self.assertEqual(callbacks.get_default_callbacks(), original_callbacks)
                self.assertIsNone(wandb.run)
                primary = rank == 0 or (rank == -1 and devices == 1)
                self.assertEqual(initialize.call_count, int(primary))
                if primary:
                    tracking, = runs
                    self.assertEqual(tracking.entity, "configured-team")
                    self.assertEqual(tracking.project, "season-2026-2027")
                    self.assertEqual(tracking.group, "ultralytics/yolo26s")
                    self.assertEqual(tracking.name, run.out_dir.name)
                    self.assertEqual(tracking.id, run.run_id)
                    self.assertTrue(tracking._is_finished)
                    self.assertEqual(tracking._exit_code, int(fails))


class ModelOutputTests(unittest.TestCase):
    def test_ddp_startup_preserves_slurm_logs_and_tracking_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.dict(os.environ, {"YOLO_CONFIG_DIR": str(root / "yolo-config")}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                from ultralytics.utils import dist

            config = base.Run(
                data_dir=root / "data",
                model_cache_dir=root / "cache",
                out_dir=root / "yolo26s",
                slurm={"cpus-per-task": 24, "gpus-per-node": 2},
                model_path=Path("ultralytics/yolo26s"),
                run_id="0123456789abcdef0123456789abcdef",
                run_timestamp="20261020T142813000000Z", data_version=1,
            )
            files = [config.out_dir / "logs/slurm-123.out", config.out_dir / "wandb/run.log"]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("active run\n")

            def prepare_ddp(**options):
                save_dir = Path(options["save_dir"])
                (save_dir / "weights").mkdir(parents=True, exist_ok=True)
                trainer = SimpleNamespace(save_dir=save_dir, resume=False, world_size=2)
                dist.generate_ddp_command(trainer)

            model = runpy.run_path(str(base.ROOT / "models/ultralytics/yolo26s.py"))
            with (
                patch("ultralytics.YOLO") as yolo,
                patch("ultralytics.settings.update"),
                patch.object(dist, "generate_ddp_file", return_value="unused-ddp.py"),
                patch.object(dist, "find_free_network_port", return_value=29500),
            ):
                yolo.return_value.train.side_effect = prepare_ddp
                base._execute(model["train"].__wrapped__, config)
                yolo.assert_called_once_with(str(config.model_cache_dir / "yolo26s.pt"))
                yolo.return_value.train.assert_called_once()

            for path in files:
                self.assertTrue(path.is_file(), f"DDP startup deleted {path}")
                self.assertEqual(path.read_text(), "active run\n")


if __name__ == "__main__":
    unittest.main()
