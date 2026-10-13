import argparse
import fcntl
import inspect
import json
import re
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class SlurmConfig:
    _slurm: dict[str, object] = field(default_factory=lambda: {
        "account": "PAS3560",
        "clusters": "pitzer",
        "partition": "gpu-exp",
        "gpus-per-node": 1,
        "cpus-per-task": 8,
        "time": "01:00:00",
        "nodes": 1,
        "ntasks": 1,
    }, repr=False)

    def __post_init__(self):
        self._slurm = self._slurm.copy()

    @property
    def slurm(self):
        return self._slurm

    @slurm.setter
    def slurm(self, updates):
        self._slurm = {**self._slurm, **updates}

    def with_options(self, *, slurm={}):
        config = replace(self)
        config.slurm = slurm
        return config


class BaseModel(ABC):
    config = SlurmConfig()
    data_version: int = 1
    model_cache_dir: Path = ROOT / ".modelcache"
    model_path: Path = Path()
    _run_number: int | None = None
    _run_timestamp: str | None = None
    _resolved_data_version: int | None = None

    @classmethod
    def _data_directories(cls):
        return {
            int(directory.name.split(".", 1)[0][1:]): directory
            for directory in (ROOT / "data").glob("v*")
            if directory.is_dir()
        }

    @property
    def data_dir(self):
        version = self._resolved_data_version if self._resolved_data_version is not None else self.data_version
        if type(version) is not int:
            raise TypeError("data_version must be an integer")
        versions = self._data_directories()
        version = max(versions, default=-1) if version == -1 else version
        return versions[version]

    @classmethod
    def get_out_dir(cls):
        if cls._run_number is None or cls._run_timestamp is None or cls._resolved_data_version is None:
            raise RuntimeError("A run must be reserved before training")
        timestamp = datetime.strptime(cls._run_timestamp, "%Y%m%dT%H%M%S%fZ").strftime("%m-%d-%H:%M-%y")
        return ROOT / "out" / cls.model_path / f"v{cls._run_number}.dv{cls._resolved_data_version}.{timestamp}"

    @property
    def out_dir(self):
        return type(self).get_out_dir()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        model_path = Path(inspect.getfile(cls)).resolve().relative_to(ROOT)
        if model_path.parts[0] == "models":
            model_path = Path(*model_path.parts[1:])
        cls.config = cls.config.with_options()
        cls.model_path = model_path.with_suffix("")
        cls._run_number = None
        cls._run_timestamp = None
        cls._resolved_data_version = None
        train = cls.__dict__.get("train")
        if train is not None and not getattr(train, "__isabstractmethod__", False):
            from tracking import tracked_train

            cls.train = tracked_train(train)

    @abstractmethod
    def train(self):
        pass

    @classmethod
    def launch(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--train",
            action="store_true",
            help=argparse.SUPPRESS,
        )
        parser.add_argument("--run", type=int, help=argparse.SUPPRESS)
        parser.add_argument("--timestamp", help=argparse.SUPPRESS)
        parser.add_argument("--data-version", type=int, help=argparse.SUPPRESS)
        args = parser.parse_args()

        if args.train:
            cls.prepare_run(args.run, args.timestamp, args.data_version)
            cls().train()
        else:
            cls.submit()

    @classmethod
    def submit(cls) -> None:
        cls.prepare_run()
        script_path = Path(inspect.getfile(cls)).resolve()
        run_command = shlex.join([
            "srun",
            sys.executable,
            str(script_path),
            "--train",
            "--run",
            str(cls._run_number),
            "--timestamp",
            cls._run_timestamp,
            "--data-version",
            str(cls._resolved_data_version),
        ])

        directives = cls.config.slurm.copy()
        directives.setdefault("job-name", cls.model_path.name.lower())
        directives.setdefault("output", cls.get_out_dir() / "logs" / "slurm-%j.out")
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

    @classmethod
    def prepare_run(cls, run=None, timestamp=None, data_version=None):
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        elif not re.fullmatch(r"\d{8}T\d{12}Z", timestamp):
            raise ValueError("Run timestamp must have the UTC format YYYYMMDDTHHMMSSffffffZ")
        data_version = cls.data_version if data_version is None else data_version
        if type(data_version) is not int:
            raise TypeError("data_version must be an integer")
        if data_version == -1:
            versions = cls._data_directories()
            if not versions:
                raise KeyError(-1)
            data_version = max(versions)
        if run is None:
            with (ROOT / "run_counts.json").open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                handle.seek(0)
                counts = json.loads(handle.read() or "{}")
                model_key = cls.model_path.as_posix()
                legacy_run_number = max(
                    (value for key, value in counts.items() if key.casefold() == cls.__name__.casefold()),
                    default=0,
                )
                previous_run_number = max(
                    (value for key, value in counts.items() if key.casefold() == model_key.casefold()),
                    default=legacy_run_number,
                )
                run = previous_run_number + 1
                counts[model_key] = run
                handle.seek(0)
                handle.truncate()
                json.dump(counts, handle, indent=2)
                handle.write("\n")
        cls._run_number = run
        cls._run_timestamp = timestamp
        cls._resolved_data_version = data_version
        (cls.get_out_dir() / "logs").mkdir(parents=True, exist_ok=True)


def configure(*, slurm={}):
    """Merge Slurm overrides into a model's independent configuration."""
    def apply(cls):
        cls.config = cls.config.with_options(slurm=slurm)
        return cls

    return apply


def slurmable(cls):
    if cls.__module__ == "__main__":
        cls.launch()
    return cls
