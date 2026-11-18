"""Dataset configuration for every MMDetection model in this directory."""

import json

from base import Run


def load(run: Run) -> dict[str, object]:
    source = run.data_dir / "coco"
    annotations = source / "annotations"
    categories = json.loads((annotations / "instances_train.json").read_text())["categories"]
    metainfo = {"classes": tuple(category["name"] for category in sorted(categories, key=lambda item: item["id"]))}
    config = {"metainfo": metainfo}
    for split in ("train", "val", "test"):
        if split == "test" and not (annotations / "instances_test.json").is_file():
            config.update(test_dataloader=None, test_evaluator=None, test_cfg=None)
            continue
        annotation = (annotations / f"instances_{split}.json").resolve(strict=True)
        images = (source / "images" / split).resolve(strict=True)
        config[f"{split}_dataloader.dataset"] = {
            "data_root": str(source),
            "ann_file": str(annotation),
            "data_prefix": {"img": str(images) + "/"},
            "metainfo": metainfo,
        }
        config[f"{split}_dataloader.num_workers"] = run.slurm["cpus-per-task"]
        if split != "train":
            config[f"{split}_evaluator.ann_file"] = str(annotation)
    return config
