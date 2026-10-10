"""OSC partition documentation for editor completion and hover hints.

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
    """Pitzer partitions; NVIDIA V100 GPUs have 16 or 32 GB VRAM each."""

    batch = "batch"
    """Mixed CPU/GPU pool; V100 GPUs have 16 or 32 GB VRAM per GPU."""
    cpu = "cpu"
    """CPU-only partition; no GPUs or GPU VRAM."""
    cpu_exp = "cpu-exp"
    """CPU-only expansion partition; no GPUs or GPU VRAM."""
    debug = "debug"
    """CPU debug pool; no GPUs currently configured. Use gpudebug for GPUs."""
    debug_cpu = "debug-cpu"
    """CPU-only debug partition; no GPUs or GPU VRAM."""
    debug_exp = "debug-exp"
    """CPU-only expansion debug partition; no GPUs or GPU VRAM."""
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
    hugemem = "hugemem"
    """CPU-only huge host-memory partition; no GPUs or GPU VRAM."""
    largemem = "largemem"
    """CPU-only large host-memory partition; no GPUs or GPU VRAM."""
    longcpu = "longcpu"
    """CPU-only long-running partition; no GPUs or GPU VRAM."""
    systems = "systems"
    """Systems pool with CPU/GPU nodes; V100 VRAM is 16 or 32 GB per GPU."""
    condo_anzenbacher_olivucci = "condo-anzenbacher-olivucci"
    """CPU-only condo partition; no GPUs currently configured."""
    condo_anzenbacher_olivucci_backfill = "condo-anzenbacher-olivucci-backfill"
    """CPU-only condo backfill partition; no GPUs currently configured."""
    condo_peterson_olivucci = "condo-peterson-olivucci"
    """CPU-only condo partition; no GPUs currently configured."""
    condo_peterson_olivucci_backfill = "condo-peterson-olivucci-backfill"
    """CPU-only condo backfill partition; no GPUs currently configured."""

    @property
    def cluster(self) -> Cluster:
        return Cluster.PITZER


class Ascend(_ClusterPartition):
    """Ascend partitions; NVIDIA A100 GPUs have 40 or 80 GB VRAM each."""

    batch = "batch"
    """Mixed A100 pool: 40 or 80 GB VRAM per GPU, depending on node type."""
    cpuonly = "cpuonly"
    """CPU-only partition; GPUs are not exposed and no GPU VRAM is available."""
    debug = "debug"
    """Mixed A100 debug pool: 40 or 80 GB VRAM per GPU."""
    debug_nextgen = "debug-nextgen"
    """Nextgen debug NVIDIA A100: 40 GB VRAM per GPU; 2 or 3 GPUs per node."""
    debug_quad = "debug-quad"
    """Quad debug NVIDIA A100: 80 GB VRAM per GPU; 4 GPUs per node."""
    gpu = "gpu"
    """No nodes currently configured; no available GPU VRAM. Use nextgen or quad."""
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
    """Cardinal partitions; NVIDIA H100 GPUs have 94 GB VRAM each.

    OSC's cluster overview lists 94 GB per GPU; its technical specifications
    page lists 96 GB. These hints use the lower overview figure for planning:
    https://www.osc.edu/resources/technical_support/supercomputers/cardinal/technical_specifications
    """

    batch = "batch"
    """Mixed CPU/GPU pool; NVIDIA H100 GPUs have 94 GB VRAM per GPU."""
    cache = "cache"
    """CPU-only HBM cache partition; host HBM is not GPU VRAM. No GPUs."""
    cpu = "cpu"
    """CPU-only partition; host HBM is not GPU VRAM. No GPUs."""
    debug = "debug"
    """Mixed CPU/GPU debug pool; NVIDIA H100 GPUs have 94 GB VRAM per GPU."""
    gpu = "gpu"
    """NVIDIA H100: 94 GB VRAM per GPU; 4 GPUs per node with NVLink."""
    hugemem = "hugemem"
    """CPU-only huge host-memory partition; no GPUs or GPU VRAM."""
    longcpu = "longcpu"
    """CPU-only long-running partition; no GPUs or GPU VRAM."""
    systems = "systems"
    """Systems CPU/GPU pool; NVIDIA H100 GPUs have 94 GB VRAM per GPU."""
    condo_ccapp = "condo-ccapp"
    """CPU-only condo partition; no GPUs currently configured."""
    condo_ccapp_backfill = "condo-ccapp-backfill"
    """CPU-only condo backfill partition; no GPUs currently configured."""
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
    """Select a partition, such as Ascend.nextgen; its cluster is automatic."""
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
