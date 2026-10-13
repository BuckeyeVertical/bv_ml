# Slurm model training

Use the existing Conda environment named `bv_ml` for setup, validation, and
training. Slurm runs the same Python executable on the compute node:

```bash
conda run -n bv_ml --no-capture-output python -m pip install -r requirements.txt
```

The PyTorch and torchvision versions are pinned to CUDA 12.6 builds for
[Pitzer's V100 GPUs](https://www.osc.edu/supercomputing/computing/pitzer/technical_specifications).
CUDA 13 has removed support for Volta GPUs; see the
[NVIDIA release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/).

Dataset versions live in `data/v<num>/` or `data/v<num>.<details>/`
(for example, `data/v1.visdrone/`). `self.data_dir` returns the version root;
each model handles the files and subdirectories inside it.
Version numbers must be unique, and dataset directory names must follow one of
those two forms. Set `data_version=1` to select `v1.visdrone` without typing
the suffix. Only integers are accepted, and the default is version `1`.
Set `data_version=-1` to select the highest available version number.
A missing version raises `KeyError`. Before a run is reserved, selection occurs
on each access to `self.data_dir`. Reservation pins the selected dataset
version for that run. See [dataset version selection](data/README.md).

A dataset can use this layout:

```text
data/v1/yolo/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

For that layout, `data.yaml` can contain the following. Replace the root and
class names with your dataset's values:

```yaml
path: /absolute/path/to/bv_ml/data/v1/yolo
train: images/train
val: images/val
names:
  0: your_class
```

Other YOLO export layouts also work if `data.yaml` points to the right image
directories. See the [Ultralytics detection dataset format](https://docs.ultralytics.com/datasets/detect/).

Configure the repository import path once, from the repository root:

```bash
conda env config vars set -n bv_ml PYTHONPATH="$PWD"
```

If `bv_ml` is already active, deactivate and activate it again to apply this
setting. Conda automatically exports this path whenever you activate `bv_ml`
or use `conda run -n bv_ml`, and restores the previous value on deactivation.

Submit the example:

```bash
conda activate bv_ml
python models/ultralytics/yolo26/small.py
```

The environment's `PYTHONPATH` makes repository imports available without
import-path setup in model files. Slurm inherits it from the submitting shell
and runs the absolute model script with the same Python executable from the
repository directory. If you move the repository, update the environment
setting to its new location. Module execution such as
`python -m models.ultralytics.yolo26.small` still works.

`ROOT` in `base.py` identifies the repository root for model paths, datasets,
output folders, caches, and tracking. Each runnable model inherits directly
from `BaseModel`:

- [base.py](base.py) handles Slurm, dataset versions, run counts, timestamps,
  and output directories.
- [Small](models/ultralytics/yolo26/small.py),
  [Medium](models/ultralytics/yolo26/medium.py), and
  [Large](models/ultralytics/yolo26/large.py) each define their training
  arguments, weights, and training method. They select `yolo26s.pt`,
  `yolo26m.pt`, and `yolo26l.pt`, respectively.
- [tracking.py](tracking.py) provides the automatic W&B training wrapper.

Submit any size with the corresponding script:

```bash
conda run -n bv_ml --no-capture-output python models/ultralytics/yolo26/small.py
conda run -n bv_ml --no-capture-output python models/ultralytics/yolo26/medium.py
conda run -n bv_ml --no-capture-output python models/ultralytics/yolo26/large.py
```

`SlurmConfig` holds only Slurm settings. Use `@configure(slurm={...})` to override
those settings; omitted dictionary keys retain their inherited values.
Every subclass receives its own config and Slurm dictionary.

Dataset selection is an ordinary class attribute: declare `data_version = 2`
inside the model class to select version 2, or `data_version = -1` for the
latest available version. `BaseModel` defaults to `data_version = 1`, and
subclasses inherit it unless they override it. Read the resolved dataset path
through `self.data_dir` and the run output path through `self.out_dir`.

`SlurmConfig` defaults to `pitzer`, `gpu-exp`, one GPU, eight CPUs per task,
one node, and one task. Each model overrides those defaults with a Pitzer
training profile for the current VisDrone dataset:

| Size | Partition | GPU VRAM | CPUs | Host RAM | Time limit | Image size | Batch | Workers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | gpu | 16 GB | 8 | 32G | 12 hours | 640 | 16 | 2 |
| medium | gpu-exp | 32 GB | 12 | 48G | 24 hours | 832 | 8 | 3 |
| large | gpu-exp | 32 GB | 16 | 64G | 48 hours | 1024 | 4 | 4 |

All three use one GPU, one node, one task, account `PAS3560`, 100 epochs,
`device=0`, mixed precision (`amp=True`), and `cache=False`. GPU 0 refers to
the GPU visible inside the Slurm allocation. `mem` requests host RAM;
GPU VRAM comes from the chosen partition.

[OSC's Pitzer batch limits](https://www.osc.edu/resources/technical_support/supercomputers/pitzer/batch_limit_rules)
document V100 16 GB GPUs in `gpu`, V100 32 GB GPUs in `gpu-exp`, node sharing
for single-node GPU jobs, and a seven-day walltime maximum for both partitions.
The live `sinfo -M pitzer -p gpu,gpu-exp` configuration also confirmed these
partition GPU types and time limits when these profiles were selected.

The current dataset has 6,471 training and 548 validation images. The larger
profiles increase resolution to retain more detail for VisDrone's small
objects, following the
[Ultralytics VisDrone guidance](https://docs.ultralytics.com/datasets/detect/visdrone/).
These profiles compare different training budgets as well as model sizes;
use the same `imgsz` for all three if you want to isolate model-size effects.

Batch sizes, CPU/RAM requests, and walltimes are starting estimates, not
measured requirements or OSC recommendations for YOLO. Check GPU utilization,
peak memory, and epoch duration on the first run, then adjust them. At 100
epochs, the allotted walltimes allow roughly 7.2/14.4/28.8 minutes per epoch
for Small/Medium/Large before startup and final validation overhead.
If GPU memory is insufficient, lower `batch` or `imgsz`; increasing `mem`
does not increase GPU VRAM.

`workers` is one quarter of `cpus-per-task`. The installed Ultralytics
detection trainer requests twice that worker count for validation and keeps
persistent loader pools, so this leaves CPU capacity for both pools and the
training process. Disabling image caching keeps host RAM use bounded without
creating an additional image cache alongside the shared dataset.

These decorators merge inherited Slurm values. On runnable size classes,
`@slurmable` stays above `@configure(slurm={...})` so the configuration is
applied before submission.

Each subclass automatically receives its own config. Its `model_path` comes
from its source file relative to the repository root, with the leading
`models/` directory and `.py` suffix removed. Class names and inheritance
no longer determine output paths. A complete model can live in one file:

```python
from base import BaseModel, configure, slurmable


@slurmable
@configure(slurm={
    "clusters": "pitzer",
    "partition": "gpu",
    "gpus-per-node": 1,
    "cpus-per-task": 8,
    "mem": "32G",
    "time": "12:00:00",
})
class Small(BaseModel):
    def train(self):
        from ultralytics import YOLO

        YOLO(str(self.model_cache_dir / "yolo26s.pt")).train(
            data=str(self.data_dir / "yolo" / "data.yaml"),
            save_dir=self.out_dir,
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,
            amp=True,
            cache=False,
            workers=self.config.slurm["cpus-per-task"] // 4,
        )
```

Mark runnable models with `@slurmable` above `@configure`. Decorators apply
from the bottom upward, so configuration must finish before `@slurmable`
launches the model. `@configure` itself never submits a job.
Save new models under `models/`, inherit directly from `BaseModel`, and put
argument setup, model construction, and training in `train()`. Importing a
model module only defines its class; executing the module submits it.
Add `--train` for direct execution. `@configure` can be omitted when the base
Slurm/dataset defaults are suitable. An experiment may also inherit a concrete
model's `train()` method while defining its own Slurm/dataset configuration
in a separate file.

Each size model passes `data` and `save_dir` directly to the library
`.train()` call, plus the size-specific settings in the table above.
Other arguments use library defaults; see the
[Ultralytics training options](https://docs.ultralytics.com/modes/train/).
`BaseModel` defaults to dataset version `1`, while `SlurmConfig` supplies these
Slurm options:
`account="PAS3560"`, `clusters="pitzer"`, `partition="gpu-exp"`,
`gpus-per-node=1`, `cpus-per-task=8`, `time="01:00:00"`, `nodes=1`, and `ntasks=1`.

Dictionary keys use Slurm's native option names without leading dashes.
There is no option whitelist or aliasing: options such as `mem`, `qos`,
`constraint`, and `mail-type` work without changes to the base code.
Slurm validates names and values when submitted.

| Dictionary entry | Generated directive |
| --- | --- |
| `"mem": "32G"` | `#SBATCH --mem=32G` |
| `"exclusive": True` | `#SBATCH --exclusive` |
| `"exclusive": False` | Omitted |
| `"gpus-per-node": None` | Omitted |
| `"mem": 0` | `#SBATCH --mem=0` |

Values containing spaces are quoted. `False` and `None` suppress inherited
options; numeric zero is passed through. `clusters` and `partition` are independent
strings. For example, to switch an inherited config to Ascend:

```python
config = Small.config.with_options(
    slurm={"clusters": "ascend", "partition": "nextgen"},
)
```

`SlurmConfig()` creates a mutable config with its own default Slurm dictionary.
Assigning `config.slurm` merges your entries with its current values through
the property setter:

```python
from base import SlurmConfig

config = SlurmConfig()
config.slurm = {"mem": "32G", "exclusive": True}
config.slurm = {"mem": "64G"}  # Keeps exclusive=True and all other settings.
```

Use the setter after construction rather than `SlurmConfig(slurm={...})`.
`config.with_options(slurm={...})` copies the config, then
applies Slurm overrides through the same setter. `@configure` also uses this
method, so parent and child configs stay independent. `dataclasses.replace`
preserves existing directives and copies the dictionary as well.

`cpus-per-task` defaults to eight in the Slurm dictionary, independently of GPU
count. Submission uses that value directly. Override it through the dictionary;
`None` or `False` suppresses the CPU directive.

Subclass creation derives `model_path` from the source file;
`@slurmable` launches the model when its module is executed. Submission fills
in the lowercase model file stem as Slurm `job-name`,
the run's `logs/slurm-%j.out` as `output`, and the repository directory as `chdir`
when those keys are absent. User-supplied values take precedence, including
`None` or `False` to suppress them. All options are rendered into the batch
script before the `srun` command. See the [sbatch manual](https://slurm.schedmd.com/sbatch.html).

Write native training keyword arguments directly in the library `.train()`
call, independently of `config.slurm`. Dataset paths, output paths, and worker
counts can reference `self.data_dir`, `self.out_dir`, and `self.config.slurm`.
Each model owns its dataset layout and native training arguments.
`BaseModel` automatically wraps every concrete
`train()` implementation with an online W&B run, including direct Python
calls. Model files do not need W&B imports or explicit context managers.
Inherited methods stay wrapped, and an override calling `super().train()`
shares one run. Return values and exceptions pass through the wrapper.

Reserve a run before manually constructing a model, and construct a new
instance for each run:

```python
Small.prepare_run()
model = Small()
model.train()
```

The normal `--train` launcher already reserves the run before calling `cls()`.
Imports, `--help`, submission, and class construction do not load weights or
resolve training data. Those operations happen inside `train()`.

For other Ultralytics model families, construct the appropriate model inside
`train()`. Classification models also need a `"data"` argument pointing
to a directory containing `train/<class>/` and `val/<class>/`.
Models must support Ultralytics' training API; see the
[training arguments](https://docs.ultralytics.com/modes/train/),
[RT-DETR example](https://docs.ultralytics.com/models/rtdetr/) and
[classification dataset format](https://docs.ultralytics.com/datasets/classify/).

The example fine-tunes pretrained `yolo26s.pt` using the
[Ultralytics YOLO26 training API](https://docs.ultralytics.com/models/yolo26/).
`BaseModel.model_cache_dir` points to the shared repository `.modelcache/`
directory. Models pass explicit paths such as
`self.model_cache_dir / "yolo26s.pt"` to Ultralytics, which downloads missing
weights there and reuses them across runs. The directory is ignored by Git
and stores weight files directly, without framework subfolders. A local
checkpoint path can be used instead. Trained checkpoints belong in the run's
output directory.

Output paths follow the model file location. For
`models/ultralytics/yolo26/small.py`, `model_path` is
`Path("ultralytics/yolo26/small")`, and `self.out_dir` looks like:

```text
out/ultralytics/yolo26/small/v2.dv1.10-13-18:45-26/
  logs/slurm-<job-id>.out
```

The run directory uses `v<run>.dv<dataset version>.MM-DD-HH:MM-YY`.
For example, `v2.dv1.10-13-18:45-26` is run 2 on dataset version 1,
reserved on October 13, 2026 at 18:45 UTC. The timestamp is generated once
when reserving the run and displayed with minute precision.
Submission passes the run number, timestamp, and resolved dataset version
to the compute-node process, so queued jobs reuse the exact reserved directory
and dataset. With `data_version = -1`, the folder records the actual latest
version selected at reservation. A newer dataset arriving while the job is
queued will be used by a future run. Direct `--train` execution reserves a
fresh run, timestamp, and dataset selection as well.

`BaseModel` owns `data_version`, the file-derived `model_path`, and the
reserved `_run_number`, `_run_timestamp`, and `_resolved_data_version`. `data_dir` and `out_dir` are computed instance
properties. Submission uses the class method `get_out_dir()` so it can access
the same path before constructing a model.

Every subclass copies its inherited Slurm config and dictionary, and starts
without a reserved run number, timestamp, or dataset selection. Reserving a run updates the model
class's runtime fields while preserving its Slurm config and declared dataset selector.
An experiment defined in
`models/ultralytics/yolo26/my_model.py` gets the path
`ultralytics/yolo26/my_model`, even if it inherits from `Small`.
Keep one runnable model per file; classes defined in the same file share a
model path and run counter.

Slurm stdout and stderr default to the run's `logs/` directory. Each model
passes the run directory to training using `save_dir=self.out_dir`.
If you override the training `save_dir` argument, Slurm logs retain their
configured location.

Counters in `run_counts.json` use the file-derived `model_path`, so files in
different directories have independent counters even when their classes have
the same name. Existing paths with different capitalization, such as
`Ultralytics/YOLO26/large`, continue their counters under the new lowercase
file-derived key. If no matching path exists, a legacy counter keyed by the
class name is used when present. Existing output directories stay at their
original locations. Failed or cancelled submissions can leave gaps.

## Weights & Biases

`BaseModel` automatically applies `tracked_train()` when a subclass defines
its `train()` method. The wrapper starts the run before training and closes it
when training returns or raises an exception. This works both through the
launcher and when calling a model's `.train()` directly after reserving a run.
The W&B project uses the repository directory's folder name (`bv_ml` here),
and groups use the file-derived model path. Every training call starts an
online run, explicitly selecting `mode="online"` regardless of `WANDB_MODE`.

The wrapper handles run metadata and lifecycle. Each framework's integration
still supplies training metrics: Ultralytics needs its W&B callbacks enabled,
while Lightly Train needs `logger_args={"wandb": {}}`. Custom training loops
can call `wandb.log(...)` directly inside `train()`; the wrapper has already
opened the run. W&B initialization errors stop training before it begins.

Set up the native integration once in `bv_ml`:

```bash
conda run -n bv_ml --no-capture-output python -m pip install wandb==0.30.0
conda run -n bv_ml --no-capture-output yolo settings wandb=True
conda run -n bv_ml --no-capture-output wandb login
```

Credentials stay in W&B login storage or `WANDB_API_KEY`, never in model files.
W&B handles authentication and run cleanup; Ultralytics handles metric, plot,
and best-checkpoint logging. This connection is intended for single-GPU jobs.

`out/ultralytics/yolo26/small/v2.dv1.10-13-18:45-26/` becomes project
`bv_ml`, group `ultralytics/yolo26/small`, and run name
`ultralytics/yolo26/small/v2.dv1.10-13-18:45-26`.
The helper reads `model_path`, model file stem, numeric `run_number`,
`run_timestamp`, and resolved `data_version` from the model instance.
The models pass arguments directly to `.train()` without a
separate argument snapshot in W&B. The helper still accepts an optional
`train_args` dictionary when a custom model needs one. W&B logs stay under the
reserved run directory; it does not upload the entire output folder. Each invocation creates a new W&B run ID.

Use `WANDB_ENTITY` to select your team. W&B credentials and entity settings
must be available when you submit the Slurm job. Model files simply implement
`train()`; no explicit `wandb_run()` call is needed.

For direct execution on a machine with a GPU available:

```bash
conda run -n bv_ml --no-capture-output python models/ultralytics/yolo26/small.py --train
```

For CPU training, set `slurm={"gpus-per-node": None, "cpus-per-task": 8}`
and pass `device="cpu"` directly in the library `.train()` call.
