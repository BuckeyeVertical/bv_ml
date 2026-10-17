import argparse
import fcntl
import inspect
import json
import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import TypeVar, overload

ROOT = Path(__file__).resolve().parent

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
    run_number: int
    run_timestamp: str
    data_version: int


def _next_run_number(model_path: Path) -> int:
    with (ROOT / "run_counts.json").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        counts = json.loads(handle.read() or "{}")
        model_key = model_path.as_posix()
        legacy_run_number = max(
            (value for key, value in counts.items() if key.casefold() == model_path.name.casefold()),
            default=0,
        )
        previous_run_number = max(
            (value for key, value in counts.items() if key.casefold() == model_key.casefold()),
            default=legacy_run_number,
        )
        run_number = previous_run_number + 1
        counts[model_key] = run_number
        handle.seek(0)
        handle.truncate()
        json.dump(counts, handle, indent=2)
        handle.write("\n")
    return run_number


def _reserve_run(
    script_path: Path,
    slurm: dict[str, object],
    data_version: int,
    *,
    run_number: int | None = None,
    run_timestamp: str | None = None,
) -> Run:
    reserved = run_number is not None and run_timestamp is not None
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
    if run_number is None:
        run_number = _next_run_number(model_path)
    out_dir = ROOT / "out" / model_path / f"v{run_number:03d}.dv{data_version:03d}.{timestamp:%m-%d-%y}"
    if reserved:
        # Jobs submitted before the naming change already have Slurm logs here.
        legacy_dir = ROOT / "out" / model_path / f"v{run_number}.dv{data_version}.{timestamp:%m-%d-%H:%M-%y}"
        if not out_dir.exists() and legacy_dir.is_dir():
            out_dir = legacy_dir
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    return Run(
        data_dir=data_dir,
        out_dir=out_dir,
        model_cache_dir=ROOT / ".modelcache",
        slurm=slurm.copy(),
        model_path=model_path,
        run_number=run_number,
        run_timestamp=run_timestamp,
        data_version=data_version,
    )


def _execute(train: Callable[[Run], _Result], config: Run) -> _Result:
    import wandb

    print(config)
    group = config.model_path.as_posix()
    with wandb.init(
        project=ROOT.name,
        mode="online",
        group=group,
        name=f"{group}/{config.out_dir.name}",
        dir=str(config.out_dir),
        config={
            "model_path": group,
            "model": config.model_path.name,
            "run_number": config.run_number,
            "run_timestamp": config.run_timestamp,
            "data_version": config.data_version,
        },
    ):
        return train(config)


def _submit(script_path: Path, run: Run) -> None:
    run_command = shlex.join([
        "srun",
        sys.executable,
        str(script_path),
        "--train",
        "--run",
        str(run.run_number),
        "--timestamp",
        run.run_timestamp,
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

    script = "\n".join([*lines, "", run_command, ""])
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
    parser.add_argument("--run", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--timestamp", help=argparse.SUPPRESS)
    parser.add_argument("--data-version", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.train:
        run = _reserve_run(
            script_path,
            slurm,
            data_version if args.data_version is None else args.data_version,
            run_number=args.run,
            run_timestamp=args.timestamp,
        )
        _execute(train, run)
    else:
        _submit(script_path, _reserve_run(script_path, slurm, data_version))


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
    """Decorate train(config); running its file submits, while --train executes it.

    Accepts bare @slurmable or native Slurm options such as time="01:00:00".
    Underscores in option names become hyphens in the generated directives.
    DATA_VERSION selects the dataset (default: -1, the latest version).
    This reserved keyword is not passed to Slurm.
    Importing is idle; calling the decorated function runs training directly.
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
            raise TypeError("A @slurmable function must take exactly one config argument")

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
