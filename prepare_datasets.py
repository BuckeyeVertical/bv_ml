"""Submit YOLO-to-COCO conversion and dataset archiving to an OSC CPU node."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory, mkdtemp

import yaml
from PIL import Image

from base import ROOT

_FORMAT_VERSION = 2
_BOUNDARY_TOLERANCE = 1e-6
_IMAGE_SUFFIXES = {
    ".avif", ".bmp", ".dng", ".heic", ".jpeg", ".jpg", ".mpo",
    ".pfm", ".png", ".tif", ".tiff", ".webp",
}
_SUMMARY_FIELDS = {
    "images", "annotations", "background_images", "missing_labels", "empty_labels",
    "skipped_zero_area_boxes", "clamped_boxes",
}


def _make_readable(path: Path) -> None:
    """Share files and directories without adding write access or following links."""
    mode = path.lstat().st_mode
    if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
        return
    permissions = stat.S_IMODE(mode)
    readable = permissions | (0o555 if stat.S_ISDIR(mode) else 0o444)
    if readable != permissions:
        path.chmod(readable)
    if stat.S_ISDIR(mode):
        for child in path.iterdir():
            _make_readable(child)


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
                image.verify()
            # Reopen after verification: reading PNG EXIF can load the pixels.
            with Image.open(path) as image:
                # Pillow applies TIFF orientation and clears its tag on load.
                if image.format == "TIFF":
                    image.load()
                width, height = image.size
                # YOLO labels use the image's displayed orientation already.
                if image.getexif().get(274) in (5, 6, 7, 8):
                    width, height = height, width
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
        if manifest["format"] != "coco-detection" or type(manifest["format_version"]) is not int:
            raise ValueError(repair)
        if manifest["format_version"] != _FORMAT_VERSION:
            raise ValueError(
                f"COCO conversion version {manifest['format_version']} is unsupported "
                f"(expected {_FORMAT_VERSION}): {destination}. "
                "Move the generated coco directory aside and rerun."
            )
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
            _make_readable(destination)
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
            _make_readable(staging)
            staging.rename(destination)
        _report(destination, manifest)
        return destination


def archive_dataset(version_dir: Path, *, force: bool = False) -> Path:
    """Prepare an atomic, reusable tar archive while preserving dataset links."""
    version_dir = Path(version_dir).absolute()
    version_name = version_dir.name
    destination = version_dir.parent / "archives" / f"{version_name}.tar"
    version_dir = version_dir.resolve(strict=True)
    if not version_dir.is_dir():
        raise ValueError(f"Not a dataset version directory: {version_dir}")
    _make_readable(version_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.chmod(stat.S_IMODE(destination.parent.stat().st_mode) | 0o555)
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists() and not force:
            if not destination.is_file():
                raise ValueError(f"Dataset archive is not a file: {destination}")
            _make_readable(destination)
            print(f"Dataset archive already exists: {destination}")
            return destination
        print(f"Archiving dataset: {version_dir}", flush=True)
        with TemporaryDirectory(prefix=f".{version_dir.name}-", dir=destination.parent) as temporary:
            archive_path = Path(temporary) / "dataset.tar"
            with tarfile.open(archive_path, "w", dereference=False) as archive:
                archive.add(version_dir, arcname=version_name)
            _make_readable(archive_path)
            archive_path.replace(destination)
    print(f"Dataset archive ready: {destination}")
    return destination


_VERSION = re.compile(r"v(\d+)(?:\.(.+))?")


def _version_name(name: str, number: int) -> str:
    match = _VERSION.fullmatch(name)
    suffix = match[2] if match else name
    return f"v{number:03d}" + (f".{suffix}" if suffix else "")


def _canonical_version(name: str) -> bool:
    match = _VERSION.fullmatch(name)
    return match is not None and _version_name(name, int(match[1])) == name


def _wrap_yolo(source: Path, destination: Path) -> None:
    """Move raw YOLO into a version root, restoring the source if publishing fails."""
    if source.is_symlink():
        raise ValueError(f"Raw YOLO must be a real directory, not a symlink: {source}")
    staging = Path(mkdtemp(prefix=".prepare-layout-", dir=destination.parent))
    try:
        source.rename(staging / "yolo")
        staging.rename(destination)
    except BaseException:
        if (staging / "yolo").exists():
            (staging / "yolo").rename(source)
        staging.rmdir()
        raise


def _extract_zip(source: Path, destination: Path) -> Path:
    """Extract one dataset, allowing only paths and links inside the staging root."""
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or re.match(r"^[A-Za-z]:", name):
                raise ValueError(f"ZIP member escapes its dataset directory: {member.filename}")
            if not relative.parts or "__MACOSX" in relative.parts or relative.name == ".DS_Store":
                continue
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"ZIP member escapes its dataset directory: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.is_dir():
                target.mkdir(exist_ok=True)
            elif stat.S_ISLNK(member.external_attr >> 16):
                link = archive.read(member).decode("utf-8")
                if Path(link).is_absolute() or not (target.parent / link).resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f"ZIP symlink escapes its dataset directory: {member.filename}")
                target.symlink_to(link)
            else:
                with archive.open(member) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)

    root = destination
    while not (root / "yolo").is_dir() and not (root / "data.yaml").is_file():
        children = [path for path in root.iterdir() if not path.name.startswith(".")]
        if len(children) != 1 or not children[0].is_dir():
            raise ValueError("ZIP must contain one dataset with yolo/ or a raw YOLO data.yaml")
        root = children[0]
    return root


def _save_imports(path: Path, imports: dict) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(imports, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_inputs(data_root: Path, summary: dict) -> list[Path]:
    manifest = data_root / ".prepared-zips.json"
    imports = json.loads(manifest.read_text()) if manifest.exists() else {}
    if not isinstance(imports, dict) or any(not isinstance(item, dict) for item in imports.values()):
        raise ValueError(f"Invalid ZIP import manifest: {manifest}")
    folders = []
    zips = []
    for path in sorted(data_root.iterdir()):
        if path.name.startswith(".") or path.name == "archives":
            continue
        if path.is_dir() and (_VERSION.fullmatch(path.name) or (path / "yolo").is_dir() or (path / "data.yaml").is_file()):
            folders.append(path)
        elif path.is_file() and path.suffix.lower() == ".zip":
            zips.append(path)

    numbers = [_VERSION.fullmatch(path.name) for path in folders]
    numbers.extend(_VERSION.fullmatch(path.stem) for path in zips)
    latest = max((int(match[1]) for match in numbers if match), default=0)
    used = set()

    def allocate(name, source=None):
        nonlocal latest
        match = _VERSION.fullmatch(name)
        number = int(match[1]) if match else latest + 1
        while True:
            destination = data_root / _version_name(name, number)
            occupied = destination != source and (destination.exists() or destination.is_symlink())
            if number not in used and not occupied:
                used.add(number)
                latest = max(latest, number)
                return destination
            number = latest + 1
            latest = number

    def record_failure(source, stage, error):
        message = f"{type(error).__name__}: {error}"
        summary["skipped"].append({"version": source.name, "stage": stage, "error": message})
        print(f"SKIPPED {source.name} during {stage}: {message}", flush=True)

    versions = []
    folders.sort(key=lambda path: (not _canonical_version(path.name), path.name))
    for source in folders:
        try:
            destination = allocate(source.name, source)
            raw = (source / "data.yaml").is_file() and not (source / "yolo").is_dir()
            if raw:
                _wrap_yolo(source, destination)
                summary["wrapped"].append(destination.name)
                print(f"Wrapped raw YOLO: {source.name} -> {destination.name}/yolo", flush=True)
            elif destination != source:
                source.rename(destination)
            if destination != source:
                summary["renamed"].append({"from": source.name, "to": destination.name})
                print(f"Renamed dataset: {source.name} -> {destination.name}", flush=True)
                for imported in imports.values():
                    if imported.get("dataset") == source.name:
                        imported["dataset"] = destination.name
                if imports:
                    _save_imports(manifest, imports)
            versions.append(destination)
        except Exception as error:  # noqa: BLE001 - Normalize each dataset independently.
            record_failure(source, "normalize", error)

    for source in zips:
        try:
            digest = _sha256(source)
            previous = imports.get(source.name, {})
            if previous.get("sha256") == digest and any(path.name == previous.get("dataset") for path in versions):
                print(f"ZIP already imported: {source.name} -> {previous['dataset']}", flush=True)
                continue
            with TemporaryDirectory(prefix=".prepare-zip-", dir=data_root) as temporary:
                extracted = Path(temporary) / "contents"
                extracted.mkdir()
                root = _extract_zip(source, extracted)
                destination = allocate(source.stem)
                imports[source.name] = {"sha256": digest, "dataset": destination.name}
                # Save the intended destination first so an interrupted publish cannot cause duplicate imports.
                _save_imports(manifest, imports)
                if (root / "data.yaml").is_file() and not (root / "yolo").is_dir():
                    _wrap_yolo(root, destination)
                    summary["wrapped"].append(destination.name)
                else:
                    root.rename(destination)
            versions.append(destination)
            summary["imported"].append({"source": source.name, "version": destination.name})
            print(f"Imported ZIP: {source.name} -> {destination.name}", flush=True)
        except Exception as error:  # noqa: BLE001 - A bad ZIP must not stop later inputs.
            record_failure(source, "extract", error)

    return sorted(versions, key=lambda path: (int(_VERSION.fullmatch(path.name)[1]), path.name))


def prepare_datasets(data_root: Path) -> dict:
    """Normalize inputs, convert, and archive versions without stopping on a bad input."""
    data_root = Path(data_root).resolve(strict=True)
    summary = {"data_root": str(data_root), "prepared": [], "skipped": [], "renamed": [], "wrapped": [], "imported": []}
    with (data_root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        versions = _prepare_inputs(data_root, summary)
        if not versions and not summary["skipped"]:
            raise ValueError(f"No dataset versions found under {data_root}")
        for version_dir in versions:
            print(f"Preparing dataset: {version_dir.name}", flush=True)
            stage = "coco"
            try:
                coco_dir = convert_yolo_to_coco(version_dir)
                stage = "archive"
                archive = archive_dataset(version_dir, force=True)
            except Exception as error:  # noqa: BLE001 - A failed version must not stop later datasets.
                message = f"{type(error).__name__}: {error}"
                summary["skipped"].append({"version": version_dir.name, "stage": stage, "error": message})
                print(f"SKIPPED {version_dir.name} during {stage}: {message}", flush=True)
                continue
            summary["prepared"].append({"version": version_dir.name, "coco_dir": str(coco_dir), "archive": str(archive)})
            print(f"READY {version_dir.name}: {archive}", flush=True)

        _make_readable(data_root)

    print(f"Preparation finished: {len(summary['prepared'])} prepared, {len(summary['skipped'])} skipped", flush=True)
    return summary


def _submit(args: argparse.Namespace) -> None:
    log_dir = ROOT / "out" / "dataset-preparation"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "srun", f"--cpus-per-task={args.cpus}", sys.executable, "-u", str(Path(__file__).resolve()),
        "--run", "--data-root", str(args.data_root),
    ]
    if args.report is not None:
        command.extend(["--report", str(args.report.resolve())])
    script = "\n".join(["#!/bin/bash", "set -euo pipefail", shlex.join(command), ""])
    result = subprocess.run(
        [
            "sbatch",
            f"--account={args.account}",
            f"--clusters={args.cluster}",
            f"--partition={args.partition}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={args.cpus}",
            f"--time={args.time}",
            "--job-name=prepare-datasets",
            f"--chdir={ROOT}",
            f"--output={log_dir / 'slurm-%j.out'}",
        ],
        input=script, text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr.strip()}")
    print(result.stdout.strip())
    print(f"Preparation logs: {log_dir / 'slurm-<job-id>.out'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Prepare datasets directly inside an existing compute allocation")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data", help="Directory containing dataset versions")
    parser.add_argument("--report", type=Path, help="JSON summary path (default: out/dataset-preparation/summary-<job-id>.json)")
    parser.add_argument("--account", default="PAS3560")
    parser.add_argument("--cluster", default="pitzer")
    parser.add_argument("--partition", default="cpu-exp")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--time", default="04:00:00")
    args = parser.parse_args(argv)
    if args.cpus < 1:
        parser.error("--cpus must be at least 1")
    try:
        args.data_root = args.data_root.resolve(strict=True)
        if not args.data_root.is_dir():
            raise ValueError(f"Not a dataset directory: {args.data_root}")
        if not args.run:
            _submit(args)
            return 0
        summary = prepare_datasets(args.data_root)
        job_id = os.environ.get("SLURM_JOB_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report = args.report or ROOT / "out" / "dataset-preparation" / f"summary-{job_id}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Preparation summary: {report}", flush=True)
        return 0 if summary["prepared"] else 1
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
