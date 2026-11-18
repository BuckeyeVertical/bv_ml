import contextlib
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import base


class RFDETRTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = base.ROOT / "models/rfdetr/rfdetr-small.py"
        self.run = base.Run(
            data_dir=self.root / "data/v002", out_dir=self.root / "run",
            model_cache_dir=self.root / "cache",
            slurm={"gpus-per-node": 1, "cpus-per-task": 2},
            model_path=Path("rfdetr/rfdetr-small"),
            run_id="0123456789abcdef0123456789abcdef",
            run_timestamp="20261020T142813000000Z", data_version=2,
            wandb_entity="configured-team",
        )
        self.run.out_dir.mkdir()

    def make_dataset(self, *, with_test=False):
        from PIL import Image

        from prepare_datasets import convert_yolo_to_coco

        staged = self.root / "scratch/v002"
        yolo = staged / "yolo"
        yolo.mkdir(parents=True)
        (yolo / "data.yaml").write_text(
            "train: images/train\nval: images/val\nnames: [person, car]\n"
            + ("test: images/test\n" if with_test else "")
        )
        for split in (("train", "val", "test") if with_test else ("train", "val")):
            images = yolo / "images" / split / "nested"
            labels = yolo / "labels" / split / "nested"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (32, 32)).save(images / "a.png")
            (labels / "a.txt").write_text("1 0.5 0.5 0.5 0.5\n")
        with contextlib.redirect_stdout(io.StringIO()):
            convert_yolo_to_coco(staged)
        return staged

    def test_import_and_submission_do_not_import_framework_or_connect_to_data(self):
        with (
            patch.object(base, "_reserve_run", return_value=self.run) as reserve,
            patch.object(base, "_submit") as submit,
            patch.dict(sys.modules, {"rfdetr": None, "wandb": None}),
        ):
            runpy.run_path(str(self.script))
            reserve.assert_not_called()
            submit.assert_not_called()
            with patch.object(sys, "argv", [str(self.script)]):
                runpy.run_path(str(self.script), run_name="__main__")
        submit.assert_called_once_with(self.script, self.run)
        self.assertFalse(self.run.data_dir.exists())

    def test_data_view_preserves_annotations_and_refreshes_staged_image_links(self):
        staged = self.make_dataset(with_test=True)
        load = runpy.run_path(str(self.script.with_name("__data__.py")))["load"]
        original = {
            path.name: path.read_bytes()
            for path in (staged / "coco/annotations").iterdir()
        }
        view = Path(load(replace(self.run, data_dir=staged)))
        for split, name in (("train", "train"), ("val", "valid"), ("test", "test")):
            data = json.loads((view / name / "_annotations.coco.json").read_text())
            self.assertEqual(data["images"][0]["file_name"], "images/nested/a.png")
            self.assertEqual((view / name / data["images"][0]["file_name"]).read_bytes(),
                             (staged / f"yolo/images/{split}/nested/a.png").read_bytes())
            self.assertEqual((staged / f"coco/annotations/instances_{split}.json").read_bytes(),
                             original[f"instances_{split}.json"])
        relocated = self.root / "scratch2/v002"
        shutil.copytree(staged, relocated, symlinks=True)
        self.assertEqual(load(replace(self.run, data_dir=relocated)), str(view))
        self.assertEqual((view / "train/images").resolve(), relocated / "yolo/images/train")
        self.assertEqual(list(view.rglob(".rfdetr-*")), [])

    def test_native_training_uses_staged_coco_data_and_finishes_tracking(self):
        import wandb
        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import WandbLogger
        from rfdetr import RFDETR, RFDETRSmall, training

        self.addCleanup(wandb.teardown)
        staged = self.make_dataset()
        model = runpy.run_path(str(self.script))["train"].__wrapped__
        original_train = RFDETR.train
        real_init = wandb.init
        environment = {
            "BV_ML_DATA_DIR": str(staged),
            "WANDB_ENTITY": "environment-team",
            "WANDB_RUN_GROUP": "environment-group",
            "WANDB_RUN_ID": "environment-id",
        }
        previous = {key: os.environ.get(key) for key in environment}

        for fails in (False, True):
            with self.subTest(training_fails=fails):
                runs = []

                def offline_init(runs=runs, **kwargs):
                    kwargs["mode"] = "offline"
                    kwargs["settings"] = wandb.Settings(
                        console="off", quiet=True, disable_git=True, finish_timeout=10,
                    )
                    tracking = real_init(**kwargs)
                    runs.append(tracking)
                    return tracking

                def fit(trainer, module, datamodule, ckpt_path, fails=fails):
                    self.assertIsNone(ckpt_path)
                    config = datamodule.train_config
                    self.assertEqual(Path(config.dataset_dir), self.run.out_dir / "data")
                    self.assertEqual((Path(config.dataset_dir) / "train/images").resolve(),
                                     (staged / "yolo/images/train").resolve())
                    self.assertEqual(config.output_dir, str(self.run.train_out_dir))
                    self.assertEqual(config.devices, 1)
                    self.assertEqual(config.num_workers, 2)
                    self.assertEqual(config.batch_size, 4)
                    self.assertEqual(config.grad_accum_steps, 4)
                    self.assertEqual(datamodule.model_config.pretrain_weights,
                                     str(self.run.model_cache_dir / "rf-detr-small.pth"))
                    datamodule.setup("fit")
                    self.assertEqual(datamodule.class_names, ["person", "car"])
                    # Load a real validation image and label through RF-DETR.
                    _, target = datamodule._dataset_val[0]
                    self.assertEqual(target["labels"].tolist(), [1])
                    logger, = [logger for logger in trainer.loggers if isinstance(logger, WandbLogger)]
                    logger.log_hyperparams({"epochs": config.epochs, "batch_size": config.batch_size})
                    logger.log_metrics({"val/mAP_50_95": 0.65}, step=1)
                    self.assertEqual(dict(logger.experiment.summary)["val/mAP_50_95"], 0.65)
                    if fails:
                        raise ValueError("training failed")

                with (
                    patch.dict(os.environ, environment),
                    patch.object(RFDETRSmall, "maybe_download_pretrain_weights"),
                    patch.object(RFDETRSmall, "get_model", return_value=SimpleNamespace()),
                    patch.object(training, "RFDETRModelModule", return_value=SimpleNamespace(model=None)),
                    patch.object(Trainer, "fit", autospec=True, side_effect=fit),
                    patch("wandb.init", side_effect=offline_init) as initialize,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    if fails:
                        with self.assertRaisesRegex(ValueError, "training failed"):
                            base._execute(model, self.run)
                    else:
                        base._execute(model, self.run)
                    for key in ("WANDB_ENTITY", "WANDB_RUN_GROUP", "WANDB_RUN_ID"):
                        self.assertEqual(os.environ[key], environment[key])
                initialize.assert_called_once()
                tracking, = runs
                self.assertEqual(tracking.entity, self.run.wandb_entity)
                self.assertEqual(tracking.project, f"season-{self.run.season}")
                self.assertEqual(tracking.group, self.run.model_path.as_posix())
                self.assertEqual(tracking.name, self.run.out_dir.name)
                self.assertEqual(tracking.id, self.run.run_id)
                self.assertTrue(tracking._is_finished)
                self.assertEqual(tracking._exit_code, int(fails))
                self.assertIs(RFDETR.train, original_train)
                self.assertIsNone(wandb.run)
                self.assertIsNone(self.run.data)
                self.assertEqual({key: os.environ.get(key) for key in environment}, previous)
        saved = json.loads((self.run.train_out_dir / "training_config.json").read_text())
        self.assertEqual(saved["class_names"], ["person", "car"])

    def test_worker_rank_does_not_start_a_wandb_run(self):
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
        from rfdetr import RFDETR

        track = runpy.run_path(str(self.script.with_name("__wandb__.py")))["track"]
        original_train = RFDETR.train
        with (
            patch.object(rank_zero_only, "rank", 1),
            patch("wandb.init") as initialize,
            patch("wandb.finish") as finish,
            patch.object(RFDETR, "train") as native_train,
        ):
            native_train.side_effect = lambda model, **kwargs: WandbLogger(
                project=kwargs["project"], name=kwargs["run"], save_dir=str(self.run.train_out_dir),
            ).log_metrics({"loss": 1.0}, step=1)
            with track(self.run):
                RFDETR.train(object())
            self.assertIs(RFDETR.train, native_train)
        initialize.assert_not_called()
        finish.assert_called_once_with(exit_code=0)
        self.assertIs(RFDETR.train, original_train)
        self.assertIsNone(wandb.run)


if __name__ == "__main__":
    unittest.main()
