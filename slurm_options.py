"""GPU-capable OSC partitions for editor completion and hover hints.

CPU-only partitions and partitions without configured GPU nodes are omitted.
VRAM below is per GPU, not host RAM or the sum across GPUs in a node.
Partition hardware was checked with sinfo on 2026-10-10. GPU capacities:
https://www.osc.edu/resources/technical_support/supercomputers/pitzer/batch_limit_rules
https://www.osc.edu/resources/technical_support/supercomputers/ascend
https://www.osc.edu/resources/technical_support/supercomputers/cardinal
"""

from enum import StrEnum
from typing import TypeAlias, TypedDict


class Cluster(StrEnum):
    PITZER = "pitzer"
    ASCEND = "ascend"
    CARDINAL = "cardinal"


class _ClusterPartition(StrEnum):
    @property
    def cluster(self) -> Cluster:
        raise NotImplementedError


class Pitzer(_ClusterPartition):
    """GPU-capable Pitzer partitions; V100 GPUs have 16 or 32 GB VRAM each."""

    batch = "batch"
    """Mixed CPU/GPU pool; V100 GPUs have 16 or 32 GB VRAM per GPU."""
    gpu = "gpu"
    """NVIDIA V100: 16 GB VRAM per GPU; 2 GPUs per node."""
    gpu_exp = "gpu-exp"
    """NVIDIA V100: 32 GB VRAM per GPU; 2 GPUs per node."""
    gpu_quad = "gpu-quad"
    """NVIDIA V100: 32 GB VRAM per GPU; 4 GPUs per node with NVLink."""
    gpubackfill = "gpubackfill"
    """Backfill NVIDIA V100: 16 GB VRAM per GPU; 2 GPUs per node."""
    gpubackfill_exp = "gpubackfill-exp"
    """Expansion backfill NVIDIA V100: 32 GB VRAM per GPU; 2 GPUs per node."""
    gpubackfill_quad = "gpubackfill-quad"
    """Quad backfill NVIDIA V100: 32 GB VRAM per GPU; 4 GPUs per node."""
    gpudebug = "gpudebug"
    """Debug NVIDIA V100: 16 GB VRAM per GPU; 2 GPUs per node."""
    gpudebug_exp = "gpudebug-exp"
    """Expansion debug NVIDIA V100: 32 GB VRAM per GPU; 2 GPUs per node."""
    gpudebug_quad = "gpudebug-quad"
    """Quad debug NVIDIA V100: 32 GB VRAM per GPU; 4 GPUs per node."""
    systems = "systems"
    """Systems pool with CPU/GPU nodes; V100 VRAM is 16 or 32 GB per GPU."""

    @property
    def cluster(self) -> Cluster:
        return Cluster.PITZER


class Ascend(_ClusterPartition):
    """GPU-capable Ascend partitions; A100 GPUs have 40 or 80 GB VRAM each."""

    batch = "batch"
    """Mixed A100 pool: 40 or 80 GB VRAM per GPU, depending on node type."""
    debug = "debug"
    """Mixed A100 debug pool: 40 or 80 GB VRAM per GPU."""
    debug_nextgen = "debug-nextgen"
    """Nextgen debug NVIDIA A100: 40 GB VRAM per GPU; 2 or 3 GPUs per node."""
    debug_quad = "debug-quad"
    """Quad debug NVIDIA A100: 80 GB VRAM per GPU; 4 GPUs per node."""
    longgpu = "longgpu"
    """Long-running NVIDIA A100: 40 GB VRAM per GPU; 2 or 3 GPUs per node."""
    nextgen = "nextgen"
    """NVIDIA A100: 40 GB VRAM per GPU; 2 or 3 GPUs per node."""
    preemptible = "preemptible"
    """Mixed preemptible A100 pool: 40 or 80 GB VRAM per GPU."""
    preemptible_nextgen = "preemptible-nextgen"
    """Preemptible NVIDIA A100: 40 GB VRAM per GPU; 2 or 3 GPUs per node."""
    preemptible_quad = "preemptible-quad"
    """Preemptible NVIDIA A100: 80 GB VRAM per GPU; 4 GPUs per node."""
    quad = "quad"
    """NVIDIA A100: 80 GB VRAM per GPU; 4 GPUs per node with NVLink."""
    systems = "systems"
    """Systems CPU/GPU pool; A100 GPUs have 40 or 80 GB VRAM per GPU."""

    @property
    def cluster(self) -> Cluster:
        return Cluster.ASCEND


class Cardinal(_ClusterPartition):
    """GPU-capable Cardinal partitions; H100 GPUs have 94 GB VRAM each.

    OSC's cluster overview lists 94 GB per GPU; its technical specifications
    page lists 96 GB. These hints use the lower overview figure for planning:
    https://www.osc.edu/resources/technical_support/supercomputers/cardinal/technical_specifications
    """

    batch = "batch"
    """Mixed CPU/GPU pool; NVIDIA H100 GPUs have 94 GB VRAM per GPU."""
    debug = "debug"
    """Mixed CPU/GPU debug pool; NVIDIA H100 GPUs have 94 GB VRAM per GPU."""
    gpu = "gpu"
    """NVIDIA H100: 94 GB VRAM per GPU; 4 GPUs per node with NVLink."""
    systems = "systems"
    """Systems CPU/GPU pool; NVIDIA H100 GPUs have 94 GB VRAM per GPU."""
    condo_haiphan = "condo-haiphan"
    """Condo NVIDIA H100: 94 GB VRAM per GPU; 4 GPUs per node."""
    condo_pingzhang = "condo-pingzhang"
    """Condo NVIDIA H100: 94 GB VRAM per GPU; 4 GPUs per node."""

    @property
    def cluster(self) -> Cluster:
        return Cluster.CARDINAL


Partition: TypeAlias = Pitzer | Ascend | Cardinal


class SlurmConfigOptions(TypedDict, total=False):
    account: str
    """Slurm billing account, such as PAS3560."""
    partition: Partition
    """Select a GPU-capable partition, such as Ascend.nextgen; cluster is automatic."""
    gpus_per_node: int
    """Number of GPUs per node; zero requests a CPU job."""
    time: str
    """Slurm wall-time limit, for example 04:00:00 or 2-00:00:00."""
    cpus_per_task: int | None
    """CPU cores per task; None chooses max(1, gpus_per_node * 8)."""
    framework: str
    """Framework directory under out/, usually set by the intermediary class."""
    job_name: str
    """Slurm job name; @slurmable sets this to the model class name."""
    nodes: int
    """Number of compute nodes."""
    ntasks: int
    """Total number of Slurm tasks."""
    data_type: str
    """Dataset subdirectory, such as yolo or classification."""
    data_version: str
    """Dataset version directory; latest selects the highest numeric version."""
    data_file: str | None
    """Dataset file, such as data.yaml; None passes the dataset directory."""
