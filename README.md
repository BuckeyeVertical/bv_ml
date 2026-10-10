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

Place your YOLO detection dataset in `data/v<num>/yolo/` or
`data/v<num>.<details>/yolo/` (for example, `data/v1.visdrone/yolo/`),
including `data.yaml`. The default selects the highest numeric version;
ties use the lexicographically greatest directory name. It resolves the version once when
training starts. See [dataset version selection](data/README.md).

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

Submit the example:

```bash
conda activate bv_ml
python -m models.ultralytics.yolo26s
```

The example also supports `python models/ultralytics/yolo26s.py` from the
repository root. Its direct-script entry point puts the repository root first
on the import path so the two `base.py` files do not collide. Slurm uses the
same Python executable, an absolute script path, and the repository working directory.

The inheritance chain is `SlurmModel` → `UltralyticsModel` → `yolo26s`:

- [base.py](base.py) handles Slurm, dataset versions, run counts and directories.
- [models/ultralytics/base.py](models/ultralytics/base.py) implements model
  construction, shared training defaults, dictionary merging and metadata.
- [models/ultralytics/yolo26s.py](models/ultralytics/yolo26s.py) selects
  the `yolo26s.pt` weights through its `get_model()` method.

Use `config = Parent.config.with_options(...)` for Slurm and dataset settings.
Editors such as VS Code/Pylance can suggest the named fields and check their
types. Omitted fields retain their inherited values, and each call creates a
separate config. Mark the child with `@slurmable`.
Save this example as
`models/ultralytics/my_model.py`:

```python
from base import Pitzer, slurmable
from models.ultralytics.base import UltralyticsModel


@slurmable
class MyModel(UltralyticsModel):
    config = UltralyticsModel.config.with_options(
        partition=Pitzer.gpu_exp,
        gpus_per_node=1,
        cpus_per_task=8,
        data_version="v2",
        time="04:00:00",
    )

    def get_model(self):
        from ultralytics import YOLO
        return YOLO("yolo26s.pt")

    def get_overrides(self):
        return {"epochs": 150, "batch": 8}
```

Submit it with `conda run -n bv_ml --no-capture-output python -m models.ultralytics.my_model`.
The explicit `@slurmable` decorator registers the subclass for Slurm.
The parent implements training.
Importing a model module only defines its class; executing the module submits it.
Add `--train` for direct execution. The subclass inherits the training method.
Omit the `config` assignment when the inherited Slurm/dataset settings are suitable.

Even `get_overrides()` returning `{}` works with the defaults: `yolo26n.pt`, 100 epochs, image
size 640, batch size 16, the latest dataset, one GPU and a one-hour time limit.
Slurm/dataset fields such as `gpus_per_node`, `time`, `data_type` and `data_version`
belong in `Parent.config.with_options(...)`. Set an explicit CPU allocation with
`cpus_per_task=8` in that call. `get_overrides()` only supplies model parameters and does not
modify the SlurmConfig object. `with_options(...)` retains all unspecified
SlurmConfig fields. Override `get_overrides()` in a child to supply its training
arguments.

Select only a partition: `Pitzer.gpu_exp`, `Ascend.nextgen`, or `Cardinal.gpu`.
Each member carries its cluster, and submission supplies `--clusters`
automatically. Import `Pitzer`, `Ascend`, and `Cardinal` from `base`; editors
suggest partition names after the dot. Hyphens in Slurm names become
underscores in Python, such as `Ascend.debug_nextgen` for `debug-nextgen`.

For example, to switch an inherited config to Ascend:

```python
from base import Ascend

config = UltralyticsModel.config.with_options(partition=Ascend.nextgen)
```

`SlurmConfig(partition=Ascend.nextgen)` also derives the cluster automatically.
`config.clusters` is a read-only property of the selected partition; overriding
other settings preserves that selection. The enums contain the partition
names returned by OSC's `sinfo` on
2026-10-10; Slurm still determines which partitions your account can use.
See the OSC batch rules for [Pitzer](https://www.osc.edu/resources/technical_support/supercomputers/pitzer/batch_limit_rules),
[Ascend](https://www.osc.edu/resources/technical_support/supercomputers/ascend/batch_limit_rules),
and [Cardinal](https://www.osc.edu/resources/technical_support/supercomputers/cardinal/batch_limit_rules).

Enum values are strings, so they work in Slurm commands and JSON. Use a
cluster's enum rather than a bare string such as `"gpu"`, which occurs on
multiple clusters. Changing the partition also changes the target cluster.
Passing `cpus_per_task=None` restores automatic CPU allocation;
`data_file=None` selects a dataset directory instead of a YAML file. Existing
`dataclasses.replace(Parent.config, ...)` calls continue to work.

`SlurmConfig(...)` also provides typed constructor fields when you want a fresh
config rather than inherited settings. `@slurmable` preserves the subclass type
for editor completion and sets `job_name` to the child class name.

SlurmConfig fields match the long Slurm options with hyphens replaced by underscores:

| Field | Slurm option |
| --- | --- |
| `account` | `--account` |
| `clusters` (derived from `partition`) | `--clusters` |
| `partition` | `--partition` |
| `job_name` | `--job-name` |
| `nodes` | `--nodes` |
| `ntasks` | `--ntasks` |
| `gpus_per_node` | `--gpus-per-node` |
| `cpus_per_task` | `--cpus-per-task` |
| `time` | `--time` |

The decorator sets `job_name` to the class name. With `cpus_per_task=None`,
`resolved_cpus_per_task` supplies `max(1, gpus_per_node * 8)` when submitting.
Dataset fields, `framework`, and internal run bookkeeping have no matching Slurm option.
See the [sbatch manual](https://slurm.schedmd.com/sbatch.html).

The parent builds `train_args` directly inside each `train()` call, including
the defaults for `data` and `save_dir`, then
unpacks `**self.get_overrides()` last and calls `model.train(**train_args)`.
Every training setting can be
replaced and additional Ultralytics options can be added. For example:

```python
overrides = {"save_dir": "/path/to/experiments/trial"}
```

`config.data` can be passed directly as `model.train(data=config.data)` or as a
dictionary entry `{"data": config.data}`. The parent supplies it automatically
at training time, so subclasses normally only set the dataset version/type.
An explicit `"data"` entry in `overrides` takes precedence and
is passed to Ultralytics without resolving the versioned dataset. Reading
`config.data` in a class body resolves `"latest"` during import; omit that entry
to retain resolution at training time.

Other trainable Ultralytics families use the same parent. For RT-DETR, use
`overrides = {"model_class": "RTDETR", "model": "rtdetr-l.pt"}`.
For classification, use
`config = replace(UltralyticsModel.config, data_type="classification", data_file=None)`
and `overrides = {"model": "yolo26n-cls.pt"}`.
This passes the dataset directory
containing `train/<class>/` and `val/<class>/`, rather than a YAML file. Models
must support Ultralytics' training API; see the
[training arguments](https://docs.ultralytics.com/modes/train/),
[RT-DETR example](https://docs.ultralytics.com/models/rtdetr/) and
[classification dataset format](https://docs.ultralytics.com/datasets/classify/).

The example fine-tunes pretrained `yolo26n.pt` using the
[Ultralytics YOLO26 training API](https://docs.ultralytics.com/models/yolo26/).
Ultralytics downloads those weights if they are not available locally; a
local checkpoint path can be used instead.

The intermediary class sets `config.framework` (for example,
`UltralyticsModel` sets `framework="ultralytics"`), and child models inherit it.
Outputs go to `out/<framework>/<model>/v<run>/`, where `<model>` is the
decorated model class name. For `yolo26s`, that is
`out/ultralytics/yolo26s/v<run>/`, including `run_info.json`,
training metrics, plots, and `weights/best.pt` and `weights/last.pt`.
Slurm stdout and stderr go in that run's `logs/` directory.
Existing runs remain in their original directories; subsequent runs use the
framework directory and continue the model's existing run count.
`run_info.json` contains `config` (all SlurmConfig dataclass fields) and `train_args`
(the merged training arguments). Paths are stored as strings.
`save_dir` points directly to the versioned run directory, without adding a
suffix. If you override `save_dir`, Ultralytics saves training artifacts to
that exact directory. Slurm logs and `run_info.json` remain in
the versioned run directory, with the effective training arguments recorded.

Run numbers are reserved at submission and reused by the training process.
Failed or cancelled submissions can leave gaps.

For direct execution on a machine with a GPU available:

```bash
conda run -n bv_ml --no-capture-output python -m models.ultralytics.yolo26s --train
```

To use CPU training, set `gpus_per_node=0` in `replace(Parent.config, ...)` before direct execution.
