import argparse
import inspect
import json
import os
import re
import runpy
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import TypeVar, overload
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
SEASON = "2026-2027"

_DEFAULT_SLURM = {
    "account": "PAS3560",
    "clusters": "pitzer",
    "partition": "gpu-exp",
    "gpus-per-node": 1,
    "cpus-per-task": 8,
    "time": "01:00:00",
    "nodes": 1,
    "ntasks": 1,
}

_Result = TypeVar("_Result")


@dataclass(frozen=True)
class Run:
    """Paths, resources, and reserved metadata for one training invocation."""

    data_dir: Path
    out_dir: Path
    model_cache_dir: Path
    slurm: dict[str, object]
    model_path: Path
    run_id: str
    run_timestamp: str
    data_version: int
    season: str = SEASON

    @property
    def train_out_dir(self) -> Path:
        """Training output directory that frameworks can recreate safely."""
        return self.out_dir / "train"


def _reserve_run(
    script_path: Path,
    slurm: dict[str, object],
    data_version: int,
    *,
    run_timestamp: str | None = None,
    season: str = SEASON,
) -> Run:
    reserved = run_timestamp is not None
    if not re.fullmatch(r"\d{4}-\d{4}", season) or int(season[5:]) != int(season[:4]) + 1:
        raise ValueError("Season must name consecutive years, such as 2026-2027")
    if run_timestamp is None:
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    elif not re.fullmatch(r"\d{8}T\d{12}Z", run_timestamp):
        raise ValueError("Run timestamp must have the UTC format YYYYMMDDTHHMMSSffffffZ")
    timestamp = datetime.strptime(run_timestamp, "%Y%m%dT%H%M%S%f%z")

    if type(data_version) is not int:
        raise TypeError("data_version must be an integer")
    versions = {
        int(directory.name.split(".", 1)[0][1:]): directory
        for directory in (ROOT / "data").glob("v*")
        if directory.is_dir()
    }
    if data_version == -1:
        data_version = max(versions, default=-1)
    data_dir = versions[data_version]

    model_path = script_path.relative_to(ROOT)
    if model_path.parts[0] == "models":
        model_path = Path(*model_path.parts[1:])
    model_path = model_path.with_suffix("")
    out_dir = ROOT / "out" / season / model_path / f"{model_path.name}.{timestamp:%y-%m-%d-%H%M%S}.dv{data_version:03d}"
    metadata = {
        "season": season,
        "model_path": model_path.as_posix(),
        "run_timestamp": run_timestamp,
        "data_version": data_version,
    }
    if reserved:
        saved = json.loads((out_dir / "run.json").read_text())
        if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Reserved run metadata does not match: {out_dir}")
        run_id = saved.get("run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError(f"Reserved run has an invalid run ID: {out_dir}")
    else:
        try:
            out_dir.mkdir(parents=True)
        except FileExistsError as error:
            raise FileExistsError(f"Run output already exists: {out_dir}. Retry after the current second.") from error
        run_id = uuid4().hex
        (out_dir / "run.json").write_text(json.dumps({**metadata, "run_id": run_id}, indent=2) + "\n")
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    return Run(
        data_dir=data_dir,
        out_dir=out_dir,
        model_cache_dir=ROOT / ".modelcache",
        slurm=slurm.copy(),
        model_path=model_path,
        run_id=run_id,
        run_timestamp=run_timestamp,
        data_version=data_version,
        season=season,
    )


def _execute(train: Callable[[Run], _Result], run: Run) -> _Result:
    staged_dir = os.environ.get("BV_ML_DATA_DIR")
    if staged_dir:
        data_dir = Path(staged_dir).resolve(strict=True)
        if not data_dir.is_dir() or data_dir.name != run.data_dir.name:
            raise ValueError(f"Staged dataset must be a directory named {run.data_dir.name}: {data_dir}")
        run = replace(run, data_dir=data_dir)

    adapter_path = Path(inspect.getfile(train)).resolve().with_name("__wandb__.py")
    if adapter_path.is_file():
        adapter = runpy.run_path(str(adapter_path))
        with adapter["track"](run):
            return train(run)
    return train(run)


def _submit(script_path: Path, run: Run) -> None:
    if str(run.slurm.get("nodes") or 1) != "1":
        raise ValueError("Dataset staging currently supports single-node jobs only")
    archive = ROOT / "data" / "archives" / f"{run.data_dir.name}.tar"
    if not archive.is_file():
        command = shlex.join([sys.executable, str(ROOT / "prepare_datasets.py")])
        raise FileNotFoundError(f"Dataset archive not found: {archive}. Submit CPU dataset preparation with: {command}")

    run_command = shlex.join([
        "srun",
        f"--cpus-per-task={run.slurm.get('cpus-per-task') or 1}",
        sys.executable,
        str(script_path),
        "--train",
        "--timestamp",
        run.run_timestamp,
        "--season",
        run.season,
        "--data-version",
        str(run.data_version),
    ])
    directives = run.slurm.copy()
    directives.setdefault("job-name", run.model_path.name.lower())
    directives.setdefault("output", run.out_dir / "logs" / "slurm-%j.out")
    directives.setdefault("chdir", ROOT)

    lines = ["#!/bin/bash"]
    for key, value in directives.items():
        if value is None or value is False:
            continue
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", key):
            raise ValueError(f"Use a Slurm option name without leading dashes: {key!r}")
        if value is True:
            lines.append(f"#SBATCH --{key}")
        else:
            value = str(value)
            if "\n" in value or "\r" in value:
                raise ValueError(f"Slurm option {key!r} must have a single-line value")
            lines.append(f"#SBATCH --{key}={shlex.quote(value)}")

    staging = [
        "set -euo pipefail",
        ': "${TMPDIR:?TMPDIR is required for dataset staging}"',
        'stage_dir=$(mktemp -d "$TMPDIR/bv-ml-data.XXXXXX")',
        'trap \'rm -rf -- "$stage_dir"\' EXIT',
        f'export BV_ML_DATA_DIR="$stage_dir"/{shlex.quote(run.data_dir.name)}',
        'printf \'Staging dataset to %s\\n\' "$BV_ML_DATA_DIR"',
        f'tar -xf {shlex.quote(str(archive))} -C "$stage_dir"',
        'test -d "$BV_ML_DATA_DIR"',
    ]
    script = "\n".join([*lines, "", *staging, run_command, ""])
    result = subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr.strip()}")
    print(result.stdout.strip())


def _launch(
    train: Callable[[Run], object],
    script_path: Path,
    slurm: dict[str, object],
    data_version: int,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timestamp", help=argparse.SUPPRESS)
    parser.add_argument("--season", default=SEASON, help=argparse.SUPPRESS)
    parser.add_argument("--data-version", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.train:
        run = _reserve_run(
            script_path,
            slurm,
            data_version if args.data_version is None else args.data_version,
            run_timestamp=args.timestamp,
            season=args.season,
        )
        _execute(train, run)
    else:
        _submit(script_path, _reserve_run(script_path, slurm, data_version, season=args.season))


@overload
def slurmable(
    function: Callable[[Run], _Result],
    /,
    *,
    DATA_VERSION: int = -1,
) -> Callable[[], _Result]: ...


@overload
def slurmable(
    function: None = None,
    /,
    *,
    DATA_VERSION: int = -1,
    **slurm: object,
) -> Callable[[Callable[[Run], _Result]], Callable[[], _Result]]: ...


def slurmable(function=None, /, *, DATA_VERSION: int = -1, **slurm):
    """Decorate train(run); running its file submits, while --train executes it.

    Accepts bare @slurmable or native Slurm options such as time="01:00:00".
    Underscores in option names become hyphens in the generated directives.
    DATA_VERSION selects the dataset (default: -1, the latest version).
    This reserved keyword is not passed to Slurm.
    Importing is idle; calling the decorated function runs training directly.
    A sibling __wandb__.py can provide a track(run) context for logging.
    """
    def decorate(train):
        if (
            not inspect.isfunction(train)
            or train.__qualname__ != train.__name__
            or inspect.iscoroutinefunction(train)
            or inspect.isgeneratorfunction(train)
            or inspect.isasyncgenfunction(train)
        ):
            raise TypeError("@slurmable requires a top-level synchronous function")
        parameters = list(inspect.signature(train).parameters.values())
        if len(parameters) != 1 or parameters[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError("A @slurmable function must take exactly one run argument")

        script_path = Path(inspect.getfile(train)).resolve()
        options = {**_DEFAULT_SLURM, **{key.replace("_", "-"): value for key, value in slurm.items()}}
        data_version = DATA_VERSION

        @wraps(train)
        def wrapped():
            return _execute(train, _reserve_run(script_path, options, data_version))

        if train.__module__ == "__main__":
            _launch(train, script_path, options, data_version)
        return wrapped

    return decorate if function is None else decorate(function)
