"""Dataset configuration for every LightlyTrain model in this directory."""

from base import Run


def load(run: Run) -> dict[str, object]:
    import yaml

    data_root = run.data_dir / "yolo"
    data = yaml.safe_load((data_root / "data.yaml").read_text())
    names = data["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))

    return {
        "format": "yolo",
        "path": str((data_root / data.get("path", ".")).resolve()),
        "train": data["train"],
        "val": data["val"],
        "names": names,
    }
