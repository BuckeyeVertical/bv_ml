import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import base


class TransformersTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = base.ROOT / "models/transformers/rtdetr-r18.py"
        self.run = base.Run(
            data_dir=self.root / "shared/v002", out_dir=self.root / "run",
            model_cache_dir=self.root / "cache",
            slurm={"gpus-per-node": 1, "cpus-per-task": 0},
            model_path=Path("transformers/rtdetr-r18"),
            run_id="0123456789abcdef0123456789abcdef",
            run_timestamp="20261020T142813000000Z", data_version=2,
            wandb_entity="configured-team",
        )
        self.run.out_dir.mkdir()

    def dataset(self):
        from PIL import Image

        staged = self.root / "scratch/v002"
        for split in ("train", "val"):
            images = staged / "coco/images" / split / "nested"
            images.mkdir(parents=True)
            annotations = staged / "coco/annotations"
            annotations.mkdir(exist_ok=True)
            records, boxes = [], []
            for index, size in enumerate(((48, 32), (32, 64), (40, 50)), start=1):
                Image.new("RGB", size).save(images / f"{index}.png")
                records.append({"id": index, "file_name": f"nested/{index}.png", "width": size[0], "height": size[1]})
                if index != 2:
                    boxes.append({
                        "id": index, "image_id": index, "category_id": 8,
                        "bbox": [size[0] / 4, size[1] / 4, size[0] / 2, size[1] / 2],
                        "area": size[0] * size[1] / 4, "iscrowd": 0,
                    })
            (annotations / f"instances_{split}.json").write_text(json.dumps({
                "images": records, "annotations": boxes,
                "categories": [{"id": 3, "name": "person"}, {"id": 8, "name": "car"}],
            }))
        return staged

    def test_import_and_submission_do_not_import_framework_or_read_data(self):
        with (
            patch.object(base, "_reserve_run", return_value=self.run) as reserve,
            patch.object(base, "_submit") as submit,
            patch.dict(sys.modules, {"transformers": None, "wandb": None}),
        ):
            runpy.run_path(str(self.script))
            reserve.assert_not_called()
            submit.assert_not_called()
            with patch.object(sys, "argv", [str(self.script)]):
                runpy.run_path(str(self.script), run_name="__main__")
        submit.assert_called_once_with(self.script, self.run)
        self.assertFalse(self.run.data_dir.exists())

    def test_coco_labels_backgrounds_and_map_use_the_native_processor(self):
        import torch
        from transformers import RTDetrImageProcessor

        staged = self.dataset()
        original = (staged / "coco/annotations/instances_train.json").read_bytes()
        load = runpy.run_path(str(self.script.with_name("__data__.py")))["load"]
        with contextlib.redirect_stdout(io.StringIO()):
            data = load(replace(self.run, data_dir=staged))
        self.assertEqual(data["labels"]["id2label"], {0: "person", 1: "car"})
        processor = RTDetrImageProcessor(size={"height": 64, "width": 64})
        inputs = data["trainer"](processor)
        dataset = inputs["eval_dataset"]
        batch = inputs["data_collator"]([dataset[0], dataset[1], dataset[2]])
        self.assertEqual(batch["pixel_values"].shape, (3, 3, 64, 64))
        self.assertEqual(batch["labels"][0]["class_labels"].tolist(), [1])
        self.assertEqual(batch["labels"][1]["boxes"].shape, (0, 4))
        # Perfect detections must score 1.0 after resizing and category remapping.
        logits = torch.tensor([[[-1000.0, 1000.0]], [[-1000.0, -1000.0]]])
        boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5, 0.5]]])
        prediction = SimpleNamespace(predictions=[(logits, boxes)], label_ids=[batch["labels"][:2]])
        metrics = inputs["compute_metrics"](prediction)
        self.assertAlmostEqual(metrics["map"], 1.0)
        self.assertAlmostEqual(metrics["map_50"], 1.0)
        self.assertEqual((staged / "coco/annotations/instances_train.json").read_bytes(), original)

    def test_real_cpu_training_evaluation_and_tracking_cleanup(self):
        import torch
        import wandb
        from transformers import (
            RTDetrConfig,
            RTDetrForObjectDetection,
            RTDetrImageProcessor,
            RTDetrResNetConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )

        self.addCleanup(wandb.teardown)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        staged = self.dataset()
        model_entry = runpy.run_path(str(self.script))["train"].__wrapped__
        original_init = Trainer.__init__
        real_init = wandb.init
        environment_keys = ("WANDB_ENTITY", "WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_RUN_ID", "WANDB_DIR")
        previous = {key: os.environ.get(key) for key in environment_keys}

        class FailAfterStep(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                raise ValueError("training failed")

        for fails in (False, True):
            with self.subTest(training_fails=fails):
                runs, models, trainers = [], [], []

                def offline_init(runs=runs, **kwargs):
                    kwargs["mode"] = "offline"
                    kwargs["settings"] = wandb.Settings(console="off", quiet=True, disable_git=True, finish_timeout=10)
                    tracking = real_init(**kwargs)
                    runs.append(tracking)
                    return tracking

                def tiny_model(checkpoint, models=models, **kwargs):
                    self.assertEqual(checkpoint, "PekingU/rtdetr_r18vd")
                    self.assertEqual(kwargs["cache_dir"], str(self.run.model_cache_dir / "huggingface"))
                    self.assertTrue(kwargs["ignore_mismatched_sizes"])
                    config = RTDetrConfig(
                        backbone_config=RTDetrResNetConfig(
                            embedding_size=8, hidden_sizes=[8, 16, 32, 64], depths=[1, 1, 1, 1],
                            layer_type="basic", out_indices=[2, 3, 4],
                        ),
                        encoder_in_channels=[16, 32, 64], encoder_hidden_dim=32,
                        encoder_ffn_dim=64, encoder_attention_heads=4,
                        d_model=32, decoder_in_channels=[32, 32, 32], decoder_ffn_dim=64,
                        decoder_attention_heads=4, decoder_layers=1, num_queries=4,
                        num_denoising=0, **{key: kwargs[key] for key in ("id2label", "label2id")},
                    )
                    model = RTDetrForObjectDetection(config)
                    models.append(model)
                    return model

                def short_arguments(**kwargs):
                    kwargs.update(
                        use_cpu=True, max_steps=1, num_train_epochs=1,
                        per_device_train_batch_size=2,
                        disable_tqdm=True, dataloader_pin_memory=False,
                        logging_steps=1, optim="adamw_torch",
                    )
                    return TrainingArguments(**kwargs)

                def collect_trainer(trainer, *args, fails=fails, trainers=trainers, **kwargs):
                    if fails:
                        kwargs["callbacks"] = [FailAfterStep()]
                    original_init(trainer, *args, **kwargs)
                    trainers.append(trainer)

                with (
                    patch.dict(os.environ, {"BV_ML_DATA_DIR": str(staged)}),
                    patch.object(Trainer, "__init__", collect_trainer),
                    patch("transformers.AutoModelForObjectDetection.from_pretrained", side_effect=tiny_model),
                    patch("transformers.AutoImageProcessor.from_pretrained",
                          return_value=RTDetrImageProcessor(size={"height": 64, "width": 64})),
                    patch("transformers.TrainingArguments", side_effect=short_arguments),
                    patch("wandb.init", side_effect=offline_init) as initialize,
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    if fails:
                        with self.assertRaisesRegex(ValueError, "training failed"):
                            base._execute(model_entry, self.run)
                    else:
                        base._execute(model_entry, self.run)
                        metrics = next(log for log in trainers[0].state.log_history if "eval_map" in log)
                        self.assertEqual(trainers[0].args.per_device_eval_batch_size, 1)
                        self.assertGreaterEqual(metrics["eval_map"], 0.0)
                        self.assertTrue((self.run.train_out_dir / "checkpoint-1/model.safetensors").is_file())
                    self.assertIs(Trainer.__init__, collect_trainer)
                initialize.assert_called_once()
                self.assertEqual(len(models), 1)
                tracking, = runs
                self.assertEqual(tracking.entity, self.run.wandb_entity)
                self.assertEqual(tracking.project, f"season-{self.run.season}")
                self.assertEqual(tracking.group, self.run.model_path.as_posix())
                self.assertEqual(tracking.name, self.run.out_dir.name)
                self.assertEqual(tracking.id, self.run.run_id)
                self.assertTrue(tracking._is_finished)
                self.assertEqual(tracking._exit_code, int(fails))
                self.assertIsNone(wandb.run)
                self.assertEqual({key: os.environ.get(key) for key in environment_keys}, previous)
        self.assertIs(Trainer.__init__, original_init)
        self.assertIsNone(self.run.data)


if __name__ == "__main__":
    unittest.main()
