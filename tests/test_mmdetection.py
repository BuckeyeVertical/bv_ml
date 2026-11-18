import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import base

_HAS_MMENGINE = find_spec("mmengine") is not None
_HAS_MMDET = find_spec("mmdet") is not None


class MMDetectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = base.ROOT / "models/mmdetection/faster-rcnn-r50-fpn.py"
        self.run = base.Run(
            data_dir=self.root / "shared/v002", out_dir=self.root / "run",
            model_cache_dir=self.root / "cache",
            slurm={"gpus-per-node": 1, "cpus-per-task": 2},
            model_path=Path("mmdetection/faster-rcnn-r50-fpn"),
            run_id="0123456789abcdef0123456789abcdef",
            run_timestamp="20261020T142813000000Z", data_version=2,
            wandb_entity="configured-team",
        )
        self.run.out_dir.mkdir()

    def dataset(self, *, with_test=False):
        staged = self.root / "scratch/v002"
        for split in (("train", "val", "test") if with_test else ("train", "val")):
            images = staged / "coco/images" / split / "nested"
            images.mkdir(parents=True, exist_ok=True)
            (images / "a.png").write_bytes(b"sample image")
            annotations = staged / "coco/annotations"
            annotations.mkdir(exist_ok=True)
            (annotations / f"instances_{split}.json").write_text(json.dumps({
                "images": [{"id": 1, "file_name": "nested/a.png", "width": 32, "height": 32}],
                "annotations": [],
                "categories": [{"id": 2, "name": "person"}, {"id": 9, "name": "car"}],
            }))
        return staged

    def test_import_and_submission_do_not_import_mmdetection_or_load_data(self):
        with (
            patch.object(base, "_reserve_run", return_value=self.run) as reserve,
            patch.object(base, "_submit") as submit,
            patch.dict(sys.modules, {"mmdet": None, "mmengine": None, "wandb": None}),
        ):
            runpy.run_path(str(self.script))
            reserve.assert_not_called()
            submit.assert_not_called()
            with patch.object(sys, "argv", [str(self.script)]):
                runpy.run_path(str(self.script), run_name="__main__")
        submit.assert_called_once_with(self.script, self.run)
        self.assertFalse(self.run.data_dir.exists())

    def test_data_adapter_uses_staged_coco_paths_without_changing_annotations(self):
        load = runpy.run_path(str(self.script.with_name("__data__.py")))["load"]
        for with_test in (False, True):
            with self.subTest(test_split=with_test):
                staged = self.dataset(with_test=with_test)
                original = (staged / "coco/annotations/instances_train.json").read_bytes()
                data = load(replace(self.run, data_dir=staged))
                for split in ("train", "val", "test"):
                    if split == "test" and not with_test:
                        self.assertIsNone(data["test_dataloader"])
                        continue
                    dataset = data[f"{split}_dataloader.dataset"]
                    self.assertEqual(dataset["metainfo"]["classes"], ("person", "car"))
                    self.assertEqual(data[f"{split}_dataloader.num_workers"], 2)
                    self.assertEqual(Path(dataset["ann_file"]),
                                     staged / f"coco/annotations/instances_{split}.json")
                    self.assertEqual((Path(dataset["data_prefix"]["img"]) / "nested/a.png").read_bytes(),
                                     b"sample image")
                self.assertEqual((staged / "coco/annotations/instances_train.json").read_bytes(), original)

    @unittest.skipUnless(_HAS_MMENGINE and _HAS_MMDET, "MMEngine and MMDetection configs are required")
    def test_packaged_model_config_and_native_wandb_backend(self):
        import numpy as np
        import wandb
        from mmengine.runner import Runner

        self.addCleanup(wandb.teardown)
        model = runpy.run_path(str(self.script))["train"].__wrapped__
        original_build_visualizer = Runner.build_visualizer
        real_init = wandb.init
        for with_test, fails in ((False, False), (True, True)):
            with self.subTest(test_split=with_test, training_fails=fails):
                staged = self.dataset(with_test=with_test)
                runs = []
                configs = []

                def offline_init(runs=runs, **kwargs):
                    kwargs["mode"] = "offline"
                    kwargs["settings"] = wandb.Settings(
                        console="off", quiet=True, disable_git=True, finish_timeout=10,
                    )
                    tracking = real_init(**kwargs)
                    runs.append(tracking)
                    return tracking

                def build_runner(cls, cfg, fails=fails, configs=configs, staged=staged):
                    configs.append(cfg)
                    self.assertEqual(cfg.model.roi_head.bbox_head.num_classes, 2)
                    self.assertEqual(cfg.train_cfg.max_epochs, 12)
                    self.assertEqual(cfg.train_dataloader.batch_size, 2)
                    self.assertEqual(cfg.train_dataloader.num_workers, 2)
                    self.assertTrue(cfg.auto_scale_lr.enable)
                    self.assertEqual(cfg.get("launcher", "none"), "none")
                    self.assertEqual(cfg.work_dir, str(self.run.train_out_dir))
                    self.assertEqual(Path(cfg.train_dataloader.dataset.ann_file),
                                     staged / "coco/annotations/instances_train.json")
                    self.assertEqual(Path(cfg.val_evaluator.ann_file),
                                     staged / "coco/annotations/instances_val.json")
                    self.assertEqual(cfg.val_dataloader.dataset.metainfo.classes, ("person", "car"))
                    if cfg.test_dataloader is None:
                        self.assertIsNone(cfg.test_evaluator)
                        self.assertIsNone(cfg.test_cfg)
                    else:
                        self.assertEqual(Path(cfg.test_dataloader.dataset.ann_file),
                                         staged / "coco/annotations/instances_test.json")
                        self.assertEqual(Path(cfg.test_evaluator.ann_file),
                                         staged / "coco/annotations/instances_test.json")
                    visualizer = Runner.build_visualizer(
                        SimpleNamespace(_experiment_name=self.root.name, _log_dir=cfg.work_dir),
                        {**cfg.visualizer, "type": "Visualizer", "name": f"{self.root.name}-{fails}"},
                    )
                    visualizer.add_config(cfg)

                    def train():
                        visualizer.add_scalars({"loss": 1.25, "coco/bbox_mAP": 0.65}, step=1)
                        visualizer.add_image("predictions", np.zeros((8, 8, 3), dtype=np.uint8), step=1)
                        self.assertEqual(dict(wandb.run.summary)["coco/bbox_mAP"], 0.65)
                        self.assertIn("predictions", dict(wandb.run.summary))
                        if fails:
                            raise ValueError("training failed")
                        visualizer.close()

                    return SimpleNamespace(train=train)

                with (
                    patch.dict(os.environ, {"BV_ML_DATA_DIR": str(staged)}),
                    patch.object(Runner, "from_cfg", classmethod(build_runner)),
                    patch("torch.hub.set_dir") as cache,
                    patch("wandb.init", side_effect=offline_init) as initialize,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    if fails:
                        with self.assertRaisesRegex(ValueError, "training failed"):
                            base._execute(model, self.run)
                    else:
                        base._execute(model, self.run)
                    self.assertIs(Runner.build_visualizer, original_build_visualizer)
                cache.assert_called_once_with(str(self.run.model_cache_dir / "hub"))
                initialize.assert_called_once()
                self.assertEqual(len(configs), 1)
                tracking, = runs
                self.assertEqual(tracking.entity, self.run.wandb_entity)
                self.assertEqual(tracking.project, f"season-{self.run.season}")
                self.assertEqual(tracking.group, self.run.model_path.as_posix())
                self.assertEqual(tracking.name, self.run.out_dir.name)
                self.assertEqual(tracking.id, self.run.run_id)
                self.assertTrue(tracking._is_finished)
                self.assertEqual(tracking._exit_code, int(fails))
                self.assertIsNone(wandb.run)
        self.assertIs(Runner.build_visualizer, original_build_visualizer)
        self.assertIsNone(self.run.data)


if __name__ == "__main__":
    unittest.main()
