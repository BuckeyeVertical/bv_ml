"""Dataset configuration for every Ultralytics model in this directory."""

from base import Run


def load(run: Run) -> str:
    return str(run.data_dir / "yolo" / "data.yaml")
