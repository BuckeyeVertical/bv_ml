"""Dataset configuration for every RF-DETR model in this directory."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from base import Run


def load(run: Run) -> str:
    source = run.data_dir / "coco"
    destination = run.out_dir / "data"
    for split, name in (("train", "train"), ("val", "valid"), ("test", "test")):
        annotations = source / "annotations" / f"instances_{split}.json"
        if split == "test" and not annotations.is_file():
            continue
        data = json.loads(annotations.read_text())
        images = (source / "images" / split).resolve(strict=True)
        for image in data["images"]:
            image["file_name"] = f"images/{image['file_name']}"

        split_dir = destination / name
        split_dir.mkdir(parents=True, exist_ok=True)
        # Workers can build the same view; publish complete metadata atomically.
        with TemporaryDirectory(prefix=".rfdetr-", dir=split_dir) as temporary:
            staging = Path(temporary)
            (staging / "images").symlink_to(images, target_is_directory=True)
            (staging / "_annotations.coco.json").write_text(json.dumps(data) + "\n")
            (staging / "images").replace(split_dir / "images")
            (staging / "_annotations.coco.json").replace(split_dir / "_annotations.coco.json")
    return str(destination)
