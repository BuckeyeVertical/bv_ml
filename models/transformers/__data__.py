"""COCO data and detection metrics for Transformers models in this directory."""

import json
from functools import partial
from types import SimpleNamespace

import torch
from PIL import ImageOps
from torchmetrics.detection import MeanAveragePrecision
from torchvision.datasets import CocoDetection
from torchvision.ops import box_convert

from base import Run


class _Dataset(CocoDetection):
    def __init__(self, source, split, label_ids):
        super().__init__(str(source / "images" / split), str(source / "annotations" / f"instances_{split}.json"))
        self.label_ids = label_ids

    def __getitem__(self, index):
        image, annotations = super().__getitem__(index)
        return {
            "image": ImageOps.exif_transpose(image),
            "target": {
                "image_id": self.ids[index],
                "annotations": [dict(annotation, category_id=self.label_ids[annotation["category_id"]])
                                for annotation in annotations],
            },
        }


def _collate(batch, *, processor):
    return processor(
        images=[sample["image"] for sample in batch],
        annotations=[sample["target"] for sample in batch],
        return_tensors="pt",
    )


def _predictions(outputs, labels):
    # DETR-family outputs can include loss_dict before logits and boxes.
    offset = int(isinstance(outputs[0], dict))
    return outputs[offset:offset + 2]


def _metrics(evaluation, *, processor):
    metric = MeanAveragePrecision(box_format="xyxy")
    for predictions, labels in zip(evaluation.predictions, evaluation.label_ids):
        sizes = torch.stack([torch.as_tensor(label["orig_size"]) for label in labels])
        outputs = SimpleNamespace(logits=torch.as_tensor(predictions[0]), pred_boxes=torch.as_tensor(predictions[1]))
        predicted = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=sizes)
        targets = []
        for label, (height, width) in zip(labels, sizes):
            boxes = box_convert(torch.as_tensor(label["boxes"]), in_fmt="cxcywh", out_fmt="xyxy")
            targets.append({
                "boxes": boxes * torch.tensor([width, height, width, height]),
                "labels": torch.as_tensor(label["class_labels"]),
            })
        metric.update(predicted, targets)
    results = metric.compute()
    return {name: results[name].item() for name in ("map", "map_50", "map_75", "mar_100")}


def load(run: Run) -> dict[str, object]:
    source = run.data_dir / "coco"
    categories = json.loads((source / "annotations/instances_train.json").read_text())["categories"]
    categories = sorted(categories, key=lambda category: category["id"])
    id2label = {index: category["name"] for index, category in enumerate(categories)}
    label_ids = {category["id"]: index for index, category in enumerate(categories)}
    train = _Dataset(source, "train", label_ids)
    validation = _Dataset(source, "val", label_ids)

    def trainer(processor):
        return {
            "train_dataset": train,
            "eval_dataset": validation,
            "processing_class": processor,
            "data_collator": partial(_collate, processor=processor),
            "compute_metrics": partial(_metrics, processor=processor),
            "preprocess_logits_for_metrics": _predictions,
        }

    return {
        "labels": {"id2label": id2label, "label2id": {name: index for index, name in id2label.items()}},
        "loader_args": {
            "dataloader_num_workers": run.slurm["cpus-per-task"],
            "remove_unused_columns": False,
            "eval_do_concat_batches": False,
        },
        "trainer": trainer,
    }
