# bv_ml

Model training on OSC with Slurm, versioned datasets, and online Weights &
Biases tracking. Define a training function under `models/` and decorate it
with `@slurmable`; the repository handles job submission and run bookkeeping.

## Get started

Run commands from the repository root using the existing Conda environment:

```bash
conda activate bv_ml
wandb login
yolo settings wandb=True
```

The environment sets `PYTHONPATH=.` so model files can import `base.py`.
W&B authentication is required for training; the `yolo settings` command
also enables Ultralytics metric logging.

Make sure `data/` points to your datasets and contains a version with a
`yolo/data.yaml` file. In this checkout, `data/` is a symlink to shared storage.
Then submit a model:

```bash
python models/ultralytics/yolo26/small.py
```

This reserves a run and submits it through `sbatch`. Training starts when
Slurm schedules the job. Submit from `bv_ml`: the job uses the same Python
executable as the submission process.

Available model files:

| Model | Entry point |
| --- | --- |
| YOLO26 small | `models/ultralytics/yolo26/small.py` |
| YOLO26 medium | `models/ultralytics/yolo26/medium.py` |
| YOLO26 large | `models/ultralytics/yolo26/large.py` |

Each file defines its own Slurm resources and training settings. Small uses
the latest dataset; medium and large currently pin `DATA_VERSION=1`.
Review the file before submission, including its wall time and batch size.

## Add a model

Create one file per model configuration under `models/`. The training
function must be a top-level synchronous function with one argument:

```python
from base import Run, slurmable


@slurmable(cpus_per_task=8, time="01:00:00")
def train(config: Run):
    from ultralytics import YOLO

    YOLO(str(config.model_cache_dir / "yolo26s.pt")).train(
        data=str(config.data_dir / "yolo" / "data.yaml"),
        save_dir=config.out_dir,
        epochs=100,
        imgsz=640,
        batch=8,
        amp=True,
        cache=False,
        workers=config.slurm["cpus-per-task"] // 4,
    )
```

The `Run` annotation is optional. Keep framework imports inside the function
so submission does not load the training framework. Set Slurm wall time to
cover the intended training run; the example's one-hour allocation is a limit,
not an estimate for 100 epochs.

Running the file submits automatically; no `if __name__ == "__main__"` block
or explicit `train()` call is needed. Other execution modes are:

| Command or action | Behavior |
| --- | --- |
| `python models/ultralytics/yolo26/small.py` | Reserve a run and submit to Slurm |
| `python -m models.ultralytics.yolo26.small` | Submit the same model as a module |
| `python models/ultralytics/yolo26/small.py --train` | Reserve a run and train in the current process |
| Import a model module | Define the function without starting training |
| Call an imported `train()` | Reserve a run and train in the current process |

Use direct execution where the required compute resources are already
allocated. Direct callers receive the training function's return value;
returning a result is optional. Exceptions propagate after W&B cleanup.

### Slurm options

Bare `@slurmable` and `@slurmable()` use these defaults:

| Option | Default |
| --- | --- |
| `account` | `PAS3560` |
| `clusters` | `pitzer` |
| `partition` | `gpu-exp` |
| `gpus_per_node` | `1` |
| `cpus_per_task` | `8` |
| `time` | `01:00:00` |
| `nodes` | `1` |
| `ntasks` | `1` |

Pass overrides directly to the decorator. Underscores become hyphens, so
`cpus_per_task=12` produces `#SBATCH --cpus-per-task=12`. Native names also
work through dictionary unpacking: `@slurmable(**{"cpus-per-task": 12})`.
`True` produces a flag, `False` or `None` omits an option, and numeric zero
is passed through. Default job names use the model file's lowercase stem;
the working directory defaults to the repository root.

`DATA_VERSION` is the reserved decorator argument for dataset selection.
Every other keyword becomes a Slurm option. Each run receives its own
Slurm dictionary, with hyphenated keys such as `config.slurm["cpus-per-task"]`.

### Run context

| Field | Meaning |
| --- | --- |
| `data_dir` | Selected dataset version directory |
| `out_dir` | Reserved output directory for this run |
| `model_cache_dir` | Shared repository `.modelcache/` directory |
| `slurm` | Merged Slurm options |
| `model_path` | Model file path relative to `models/`, without `.py` |
| `run_number` | Integer run number for this model path |
| `run_timestamp` | Full UTC reservation timestamp |
| `data_version` | Resolved integer dataset version |

## Datasets

Keep one YOLO source per version, using names such as `data/v001.visdrone/`.
Existing unpadded names such as `v1.visdrone` also work. Use one directory
per numeric version; avoid keeping both `v1.*` and `v001.*`.
Dataset directories are supplied externally; importing and uploading are
not implemented.

The decorator defaults to `DATA_VERSION=-1`, selecting the highest numeric
version. Pin a dataset with a decorator argument:

```python
@slurmable(DATA_VERSION=1, cpus_per_task=12)
def train(config):
    ...
```

Use integer `1`, even when the directory is named `v001.visdrone`.
Module-level `DATA_VERSION` variables are not read. Submission pins the
selected version, so a newer dataset arriving while a job is queued does
not change that run.

### Convert YOLO to COCO

Generate a COCO detection view explicitly before training a model that
needs it:

```bash
python dataset.py convert --version 1 --to coco
```

Use `--version -1` for the latest dataset. The converter creates `coco/`
beside `yolo/`, so it needs write access to the dataset's storage. Images
remain in YOLO; COCO accesses them through relative directory symlinks.
For a VisDrone version, the layout is:

```text
data/v001.visdrone/
├── yolo/
│   ├── data.yaml
│   ├── VisDrone2019-DET-train/
│   ├── VisDrone2019-DET-val/
│   └── VisDrone2019-DET-test-dev/
└── coco/
    ├── images/
    │   ├── train -> ../../yolo/VisDrone2019-DET-train/images
    │   ├── val   -> ../../yolo/VisDrone2019-DET-val/images
    │   └── test  -> ../../yolo/VisDrone2019-DET-test-dev/images
    ├── annotations/
    │   ├── instances_train.json
    │   ├── instances_val.json
    │   └── instances_test.json
    └── manifest.json
```

Use `config.data_dir / "coco/images/train"` and
`config.data_dir / "coco/annotations/instances_train.json"` in the framework's
training API. Python callers can use
`dataset.convert_yolo_to_coco(version_dir)`, which returns the `coco/` path.

The converter reads `names`, optional `path`, and declared `train`, `val`,
and `test` splits from `yolo/data.yaml`. Each split must point to one image
directory under `yolo/`. Standard `images/train` / `labels/train` and
VisDrone `split/images` / `split/labels` layouts are supported, including
nested filenames. Only declared splits are exported; image-list files and
lists of directories are unsupported.

Conversion supports bounding boxes. It maps YOLO class IDs to COCO IDs by
adding one and writes pixel `[x, y, width, height]` boxes, area, and
`iscrowd=0`. Missing or empty labels produce background images. Zero-area
boxes are skipped and reported; boundary overflow up to `1e-6` is clamped.
Malformed labels, larger overflow, unknown classes, nonfinite coordinates,
and unreadable images stop conversion. Masks and keypoints are unsupported.

Rerunning verifies the source fingerprint and output hashes, then reuses
matching output. Source changes require a new dataset version. Damaged or
unexpected COCO output is preserved and reported for review. Conversion
uses a per-version lock and temporary output directory so concurrent or
failed conversions do not publish partial results. Moving a complete
version preserves its image links when YAML paths are relative.

## Outputs and tracking

The model's file path determines its output location:

```text
models/ultralytics/yolo26/small.py
    → out/ultralytics/yolo26/small/v017.dv001.10-17-26/
```

The folder format is `v<run>.dv<dataset>.MM-DD-YY`. Both version numbers
have at least three digits; values above 999 expand normally. Dates use
UTC, and the full timestamp remains available in run metadata. Counters
in `run_counts.json` advance independently per model path and retain
compatibility with legacy counters.

Slurm logs live under `<run>/logs/slurm-<job-id>.out`. The training example
passes `config.out_dir` to YOLO for its results and puts the requested model
weights in `.modelcache/`. Other frameworks should receive these paths
through their own APIs. Existing folders keep their names; jobs submitted
with the previous naming format reuse their reserved folders.

Every training invocation starts an online W&B run:

| W&B field | Value |
| --- | --- |
| Project | Repository directory name (`bv_ml`) |
| Group | Model path, such as `ultralytics/yolo26/small` |
| Run name | Model path followed by the output folder name |
| Local files | `<run>/wandb/` |
| Config | Model path/name, run number, timestamp, dataset version |

Framework metrics require the framework's own W&B integration. Wrapping a
function with `@slurmable` creates the run context; custom loops must still
log their metrics.

## Environment and checks

All commands above assume `conda activate bv_ml`. Without activation, use
an explicit environment selection:

```bash
conda run -n bv_ml --no-capture-output python models/ultralytics/yolo26/small.py
```

To apply `environment.yml` to the existing environment, then refresh its
environment variables:

```bash
conda env update -n bv_ml -f environment.yml
conda deactivate
conda activate bv_ml
```

The YAML records the installed environment, including Conda and pip
packages. To update that snapshot from the active environment:

```bash
conda env export > environment.yml
```

`PYTHONPATH=.` follows the current working directory, so run from the
repository root. If you override Slurm's `chdir`, provide an import path
that works from that directory.

Run the tests and lint checks from `bv_ml`:

```bash
python -m unittest discover -s tests -v
python -m ruff check base.py dataset.py tests
```

Tests use temporary datasets and mocked submission/tracking; they do not
submit Slurm jobs or launch model training.
