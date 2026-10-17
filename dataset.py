import argparse
import fcntl
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from PIL import Image

from base import ROOT

_FORMAT_VERSION = 1
_BOUNDARY_TOLERANCE = 1e-6
_IMAGE_SUFFIXES = {
    ".avif", ".bmp", ".dng", ".heic", ".jpeg", ".jpg", ".mpo",
    ".pfm", ".png", ".tif", ".tiff", ".webp",
}
_SUMMARY_FIELDS = {
    "images", "annotations", "background_images", "missing_labels", "empty_labels",
    "skipped_zero_area_boxes", "clamped_boxes",
}


@dataclass
class _Split:
    name: str
    image_dir: Path
    label_dir: Path
    images: list[Path]

    def label_for(self, image: Path) -> Path:
        return (self.label_dir / image.relative_to(self.image_dir)).with_suffix(".txt")


def _images(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.rglob("*") if path.suffix.lower() in _IMAGE_SUFFIXES and path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )


def _read_source(yolo: Path) -> tuple[list[dict], list[_Split], bytes]:
    yaml_path = yolo / "data.yaml"
    yaml_bytes = yaml_path.read_bytes()
    config = yaml.safe_load(yaml_bytes)
    if not isinstance(config, dict):
        raise TypeError(f"{yaml_path}: expected a YAML mapping")
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    elif isinstance(names, dict):
        indexed = {}
        for key, name in names.items():
            if type(key) is not int and not (isinstance(key, str) and key.isdecimal()):
                raise ValueError(f"{yaml_path}: class IDs must be integers starting at zero")
            index = int(key)
            if index in indexed:
                raise ValueError(f"{yaml_path}: duplicate class ID {index}")
            indexed[index] = name
        names = indexed
    if not names or not isinstance(names, dict) or sorted(names) != list(range(len(names))):
        raise ValueError(f"{yaml_path}: names must be a nonempty list or contiguous zero-based mapping")
    if any(not isinstance(name, str) or not name.strip() for name in names.values()):
        raise ValueError(f"{yaml_path}: class names must be nonempty strings")
    if "nc" in config and (type(config["nc"]) is not int or config["nc"] != len(names)):
        raise ValueError(f"{yaml_path}: nc does not match names")
    categories = [
        {"id": index + 1, "name": names[index], "supercategory": "object"}
        for index in range(len(names))
    ]

    dataset_root = config.get("path", ".")
    if not isinstance(dataset_root, str):
        raise TypeError(f"{yaml_path}: path must be a directory path")
    dataset_root = (yolo / dataset_root).resolve()
    splits = []
    for name in ("train", "val", "test"):
        value = config.get(name)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise TypeError(f"{yaml_path}: {name} must name one image directory; lists and image-list files are unsupported")
        image_dir = (dataset_root / value).resolve()
        if not image_dir.is_dir():
            raise ValueError(f"{yaml_path}: {name} is not an image directory: {image_dir}")
        if not image_dir.is_relative_to(yolo):
            raise ValueError(f"{yaml_path}: {name} images must be stored under {yolo}")
        parts = image_dir.relative_to(yolo).parts
        if "images" not in parts:
            raise ValueError(f"{yaml_path}: {name} image path must contain an 'images' directory")
        index = len(parts) - 1 - parts[::-1].index("images")
        label_dir = yolo.joinpath(*parts[:index], "labels", *parts[index + 1:])
        images = _images(image_dir)
        if not images:
            raise ValueError(f"{yaml_path}: {name} contains no supported images: {image_dir}")
        splits.append(_Split(name, image_dir, label_dir, images))
    if not splits:
        raise ValueError(f"{yaml_path}: declare at least one of train, val, or test")
    return categories, splits, yaml_bytes


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _stamp(path: Path) -> tuple | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file():
        raise ValueError(f"Expected a source file: {path}")
    return stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev


def _snapshot(yolo: Path, splits: list[_Split], yaml_bytes: bytes) -> tuple[str, dict]:
    files = {yolo / "data.yaml"}
    for split in splits:
        files.update(split.images)
        files.update(split.label_for(image) for image in split.images)
    digest = hashlib.sha256()
    stamps = {}
    for path in sorted(files):
        before = _stamp(path)
        content_hash = None if before is None else _sha256(path)
        if path == yolo / "data.yaml" and content_hash != hashlib.sha256(yaml_bytes).hexdigest():
            raise ValueError("YOLO source changed during conversion; retry with a fixed source version")
        if _stamp(path) != before:
            raise ValueError(f"YOLO source changed during conversion: {path}")
        stamps[path] = before
        record = [path.relative_to(yolo).as_posix(), content_hash]
        digest.update(json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n")
    return digest.hexdigest(), stamps


def _assert_unchanged(stamps: dict, splits: list[_Split]) -> None:
    for path, stamp in stamps.items():
        if _stamp(path) != stamp:
            raise ValueError(f"YOLO source changed during conversion: {path}")
    for split in splits:
        if _images(split.image_dir) != split.images:
            raise ValueError(f"YOLO image inventory changed during conversion: {split.image_dir}")


def _convert_split(yolo: Path, split: _Split, categories: list[dict]) -> tuple[dict, dict, list[dict]]:
    images, annotations, issues = [], [], []
    summary = {
        "images": len(split.images), "annotations": 0, "background_images": 0,
        "missing_labels": 0, "empty_labels": 0, "skipped_zero_area_boxes": 0, "clamped_boxes": 0,
    }
    for image_id, path in enumerate(split.images, start=1):
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
        except (OSError, SyntaxError, ValueError) as error:
            raise ValueError(f"Unreadable image {path}: {error}") from error
        images.append({
            "id": image_id, "file_name": path.relative_to(split.image_dir).as_posix(),
            "width": width, "height": height,
        })
        label = split.label_for(path)
        if label.exists():
            try:
                rows = label.read_text(encoding="utf-8").splitlines()
            except UnicodeError as error:
                raise ValueError(f"{label}: labels must be UTF-8 text") from error
            if not any(row.strip() for row in rows):
                summary["empty_labels"] += 1
        else:
            rows = []
            summary["missing_labels"] += 1
        annotation_count = len(annotations)
        for line_number, row in enumerate(rows, start=1):
            if not row.strip():
                continue
            location = f"{label}:{line_number}"
            fields = row.split()
            if len(fields) != 5:
                raise ValueError(f"{location}: expected five fields: class x_center y_center width height")
            try:
                class_id, x, y, box_width, box_height = map(float, fields)
            except ValueError as error:
                raise ValueError(f"{location}: fields must be numeric") from error
            if not all(math.isfinite(value) for value in (class_id, x, y, box_width, box_height)):
                raise ValueError(f"{location}: fields must be finite")
            if not class_id.is_integer() or not 0 <= class_id < len(categories):
                raise ValueError(f"{location}: unknown class ID {fields[0]}")
            if box_width < 0 or box_height < 0:
                raise ValueError(f"{location}: negative box dimensions")
            corners = (x - box_width / 2, y - box_height / 2, x + box_width / 2, y + box_height / 2)
            if any(value < -_BOUNDARY_TOLERANCE or value > 1 + _BOUNDARY_TOLERANCE for value in corners):
                raise ValueError(f"{location}: bounding box exceeds image bounds by more than {_BOUNDARY_TOLERANCE:g}")
            left, top, right, bottom = (min(1.0, max(0.0, value)) for value in corners)
            if box_width == 0 or box_height == 0 or right <= left or bottom <= top:
                issues.append({"file": label.relative_to(yolo.parent).as_posix(), "line": line_number, "reason": "zero-area box"})
                summary["skipped_zero_area_boxes"] += 1
                continue
            if (left, top, right, bottom) != corners:
                summary["clamped_boxes"] += 1
            bbox = [left * width, top * height, (right - left) * width, (bottom - top) * height]
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": int(class_id) + 1,
                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
            })
        if len(annotations) == annotation_count:
            summary["background_images"] += 1
    summary["annotations"] = len(annotations)
    return {
        "info": {"description": "YOLO to COCO bounding-box conversion"},
        "licenses": [], "images": images, "annotations": annotations, "categories": categories,
    }, summary, issues


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")


def _existing_manifest(destination: Path, fingerprint: str, splits: list[_Split]) -> dict:
    repair = f"COCO output is unexpected or damaged: {destination}. Move the generated coco directory aside and rerun; review source changes before rebuilding."
    try:
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError(repair)
        for name in ("images", "annotations"):
            if (destination / name).is_symlink() or not (destination / name).is_dir():
                raise ValueError(repair)
        if (destination / "manifest.json").is_symlink():
            raise ValueError(repair)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format"] != "coco-detection" or type(manifest["format_version"]) is not int or manifest["format_version"] != _FORMAT_VERSION:
            raise ValueError(repair)
        source_fingerprint = manifest["source_fingerprint"]
        if not isinstance(source_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", source_fingerprint):
            raise ValueError(repair)
        if source_fingerprint != fingerprint:
            raise ValueError(f"YOLO source changed since COCO was generated for {destination.parent.name}; create a new dataset version")
        expected = {f"annotations/instances_{split.name}.json" for split in splits}
        if not isinstance(manifest["annotations"], dict) or not isinstance(manifest["summary"], dict):
            raise TypeError(repair)
        split_names = {split.name for split in splits}
        if set(manifest["annotations"]) != expected or set(manifest["summary"]) != split_names:
            raise ValueError(repair)
        if {path.name for path in (destination / "images").iterdir()} != split_names:
            raise ValueError(repair)
        if {path.name for path in (destination / "annotations").iterdir()} != {Path(name).name for name in expected}:
            raise ValueError(repair)
        for summary in manifest["summary"].values():
            if not isinstance(summary, dict) or set(summary) != _SUMMARY_FIELDS:
                raise ValueError(repair)
            if any(type(value) is not int or value < 0 for value in summary.values()):
                raise ValueError(repair)
        if not isinstance(manifest["issues"], list):
            raise TypeError(repair)
        for issue in manifest["issues"]:
            if not isinstance(issue, dict) or not isinstance(issue.get("file"), str):
                raise TypeError(repair)
            if type(issue.get("line")) is not int or issue["line"] < 1 or issue.get("reason") != "zero-area box":
                raise ValueError(repair)
        for relative, digest in manifest["annotations"].items():
            path = destination / relative
            if path.is_symlink() or _sha256(path) != digest:
                raise ValueError(repair)
        for split in splits:
            link = destination / "images" / split.name
            if not link.is_symlink() or os.path.isabs(os.readlink(link)) or link.resolve(strict=True) != split.image_dir:
                raise ValueError(repair)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(repair) from error
    return manifest


def _report(destination: Path, manifest: dict, unchanged: bool = False) -> None:
    print(f"COCO {'already up to date' if unchanged else 'generated'}: {destination}")
    for name, summary in manifest["summary"].items():
        print(
            f"  {name}: {summary['images']} images, {summary['annotations']} annotations, "
            f"{summary['background_images']} backgrounds "
            f"({summary['missing_labels']} missing / {summary['empty_labels']} empty label files), "
            f"{summary['skipped_zero_area_boxes']} skipped zero-area boxes, {summary['clamped_boxes']} clamped boxes"
        )
    for issue in manifest["issues"]:
        print(f"  {destination.parent / issue['file']}:{issue['line']}: skipped {issue['reason']}")


def convert_yolo_to_coco(version_dir: Path) -> Path:
    """Create a COCO view without copying images or modifying YOLO source files.

    Source changes after the first conversion require a new dataset version.
    Existing valid output is verified and reused; unexpected output is preserved.
    """
    version_dir = Path(version_dir).resolve(strict=True)
    if not version_dir.is_dir():
        raise ValueError(f"Not a dataset version directory: {version_dir}")
    yolo = version_dir / "yolo"
    destination = version_dir / "coco"
    with (version_dir / ".coco.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        categories, splits, yaml_bytes = _read_source(yolo)
        print(f"Checking YOLO source: {yolo}", flush=True)
        fingerprint, stamps = _snapshot(yolo, splits, yaml_bytes)
        if destination.exists() or destination.is_symlink():
            manifest = _existing_manifest(destination, fingerprint, splits)
            _assert_unchanged(stamps, splits)
            _report(destination, manifest, unchanged=True)
            return destination

        manifest = {
            "format": "coco-detection", "format_version": _FORMAT_VERSION,
            "source_fingerprint": fingerprint, "annotations": {}, "summary": {}, "issues": [],
        }
        with TemporaryDirectory(prefix=".coco-", dir=version_dir) as temporary:
            staging = Path(temporary)
            (staging / "images").mkdir()
            (staging / "annotations").mkdir()
            for split in splits:
                print(f"Converting {split.name}: {len(split.images)} images", flush=True)
                converted, summary, issues = _convert_split(yolo, split, categories)
                relative = f"annotations/instances_{split.name}.json"
                _write_json(staging / relative, converted)
                manifest["annotations"][relative] = _sha256(staging / relative)
                manifest["summary"][split.name] = summary
                manifest["issues"].extend(issues)
                target = os.path.relpath(split.image_dir, start=destination / "images")
                (staging / "images" / split.name).symlink_to(target, target_is_directory=True)
            _write_json(staging / "manifest.json", manifest)
            _assert_unchanged(stamps, splits)
            staging.rename(destination)
        _report(destination, manifest)
        return destination


def _resolve_version(version: int) -> Path:
    versions = {}
    for path in sorted((ROOT / "data").iterdir()):
        match = re.fullmatch(r"v(\d+)(?:\..+)?", path.name)
        if match and path.is_dir():
            versions.setdefault(int(match[1]), []).append(path)
    if version == -1:
        version = max(versions, default=-1)
    matches = versions.get(version, [])
    if not matches:
        raise ValueError(f"Dataset version {version} not found under {ROOT / 'data'}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous dataset version {version}: {', '.join(path.name for path in matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="Generate a derived format from a version's YOLO source")
    convert.add_argument("--version", type=int, required=True, help="Dataset version, or -1 for latest")
    convert.add_argument("--to", choices=["coco"], required=True, help="Output format")
    args = parser.parse_args()
    try:
        convert_yolo_to_coco(_resolve_version(args.version))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
