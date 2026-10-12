"""Shared configuration and Slurm entry points for training models."""
import argparse
import fcntl
import inspect
import json
import re
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar, Unpack

# Partition enums are also public imports from this module.
from slurm_options import (  # noqa: F401
    Ascend,
    Cardinal,
    Partition,
    Pitzer,
    SlurmConfigOptions,
)

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    account: str = "PAS3560"
    cluster: str = "pitzer"
    partition: Partition = Pitzer.gpu_exp
    gpus: int = 1
    time_limit: str = "01:00:00"
    _cpus: int | None = None
    framework: str = "models"
    job_name: str = "model"
    data_type: str = "yolo"
    data_version: str = "latest"
    data_file: str | None = "data.yaml"
    nodes: int = 1
    ntasks: int = 1
    _run: int | None = None

    @property
    def base_dir(self) -> Path:
        return ROOT

    @property
    def cpus(self):
        return self._cpus if self._cpus is not None else max(1, self.gpus * 8)

    def with_options(self, **options: Unpack[SlurmConfigOptions]) -> "Config":
        """Return an independent config, accepting existing model option names."""
        aliases = {
            "gpus_per_node": "gpus",
            "cpus_per_task": "_cpus",
            "time": "time_limit",
        }
        changes: dict[str, Any] = {aliases.get(key, key): value for key, value in options.items()}
        if "partition" in options:
            changes["cluster"] = options["partition"].cluster
        return replace(self, **changes)

    @property
    def data(self):
        data_root = ROOT / "data"
        version = self.data_version
        if version == "latest":
            candidates = []
            for directory in data_root.iterdir():
                match = re.fullmatch(r"v(\d+)(?:\..+)?", directory.name)
                if directory.is_dir() and match:
                    candidates.append((int(match.group(1)), directory.name))
            if not candidates:
                raise FileNotFoundError(f"No versioned datasets in {data_root}")
            version = max(candidates)[1]
        path = data_root / version / self.data_type
        return str(path / self.data_file if self.data_file else path)

    @property
    def out_dir(self):
        if self._run is None:
            raise RuntimeError("A run must be reserved before training")
        return ROOT / "out" / self.framework / self.job_name / f"v{self._run}"


class SlurmJob(ABC):
    config = Config()

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
        args = parser.parse_args()

        if args.train:
            cls.prepare_run(args.run)
            cls().train()
        else:
            cls.submit()

    @classmethod
    def submit(cls) -> None:
        cls.prepare_run()
        settings = cls.config

        script_path = Path(inspect.getfile(cls)).resolve()
        run_command = shlex.join([
            "srun",
            sys.executable,
            str(script_path),
            "--train",
            "--run",
            str(settings._run),
        ])

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={cls.__name__.lower()}",
            f"#SBATCH --nodes={settings.nodes}",
            f"#SBATCH --ntasks={settings.ntasks}",
            f"#SBATCH --cpus-per-task={settings.cpus}",
            f"#SBATCH --time={settings.time_limit}",
            f"#SBATCH --output={settings.out_dir / 'logs' / 'slurm-%j.out'}",
            f"#SBATCH --partition={settings.partition}",
            f"#SBATCH --account={settings.account}",
        ]
        if settings.gpus:
            lines.append(f"#SBATCH --gpus-per-node={settings.gpus}")

        script = "\n".join([*lines, "", run_command, ""])

        sbatch_command = ["sbatch", f"--chdir={ROOT}"]
        if settings.cluster:
            sbatch_command.append(f"--clusters={settings.cluster}")

        result = subprocess.run(
            sbatch_command,
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed:\n{result.stderr.strip()}")

        print(result.stdout.strip())

    @classmethod
    def prepare_run(cls, run=None):
        if run is None:
            with (ROOT / "run_counts.json").open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                handle.seek(0)
                counts = json.loads(handle.read() or "{}")
                run = counts.get(cls.config.job_name, 0) + 1
                counts[cls.config.job_name] = run
                handle.seek(0)
                handle.truncate()
                json.dump(counts, handle, indent=2)
                handle.write("\n")
        cls.config = replace(cls.config, _run=run)
        (cls.config.out_dir / "logs").mkdir(parents=True, exist_ok=True)


# Existing model subclasses use this name for the shared Slurm parent.
SlurmModel = SlurmJob

_ModelType = TypeVar("_ModelType", bound=SlurmJob)


def configure(**options: Unpack[SlurmConfigOptions]) -> Callable[[type[_ModelType]], type[_ModelType]]:
    """Override inherited Slurm and dataset settings without launching the class.

    Every parameter is optional. Omitted parameters retain the parent class's
    values. Supplied values, including None, replace those inherited values.
    Put @configure below @slurmable so settings apply before launch.

    Args:
        account (str): Slurm billing account, for example "PAS3560".
        partition (Partition): GPU-capable partition enum, such as
            Pitzer.gpu_exp or Ascend.nextgen. Also selects the target cluster.
        gpus_per_node (int): GPUs requested per node. Zero omits the GPU request.
        time (str): Wall-time limit, such as "04:00:00" or "2-00:00:00".
        cpus_per_task (int | None): CPU cores per task. None restores automatic
            allocation: max(1, gpus_per_node * 8).
        framework (str): Framework directory under out/, such as "ultralytics".
            Normally declared by the framework class.
        job_name (str): Slurm job name. @slurmable replaces this with the
            concrete model class name.
        nodes (int): Number of compute nodes requested.
        ntasks (int): Total number of Slurm tasks requested.
        data_type (str): Dataset format subdirectory, such as "yolo" or
            "classification". Normally declared by the framework class.
        data_version (str): Dataset version directory, such as "v2".
            "latest" selects the highest numeric version when data is resolved.
        data_file (str | None): Dataset file within the format directory,
            such as "data.yaml". None passes the directory itself to training.

    Returns:
        A decorator that assigns an independent config and returns the
        original class. The parent class's config remains unchanged.

    Example:
        @slurmable
        @configure(partition=Ascend.nextgen, time="04:00:00", data_version="v2")
        class MyModel(UltralyticsModel):
            ...
    """
    def apply(cls: type[_ModelType]) -> type[_ModelType]:
        cls.config = cls.config.with_options(**options)
        return cls

    return apply


def slurmable(cls: type[_ModelType]) -> type[_ModelType]:
    cls.config = replace(cls.config, job_name=cls.__name__)
    if cls.__module__ == "__main__":
        cls.launch()

    return cls
