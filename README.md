# bv_ml

Train models on OSC through Slurm, with shared datasets and Weights & Biases
(W&B) tracking.

Start on an OSC login node with Conda available. You need access to the GitHub
repository, OSC project `PAS3560`, and the `BuckeyeVertical` W&B team.

## Setup

### 1. Clone the repository

Use your GitHub SSH key:

```bash
git clone git@github.com:BuckeyeVertical/bv_ml.git
cd bv_ml
```

### 2. Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate bv_ml
```

The environment sets `PYTHONPATH=.`. Run the commands below from the repository
root with this environment active.

To add RF-DETR to an existing environment, run:

```bash
python -m pip install 'rfdetr[train]==1.11.2'
```

MMDetection uses a separate environment because its MMCV 2.1.0 wheel targets
PyTorch 2.1.0 and CUDA 12.1. Create it from the repository root:

```bash
conda env create -f environment.mmdetection.yml
conda activate bv_ml_mmdetection
```

The pins follow the official [MMDetection compatibility table](https://mmdetection.readthedocs.io/en/latest/notes/faq.html)
and [MMCV wheel index](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html).

### 3. Link the shared datasets

In your checkout, link the shared dataset directory:

```bash
ln -s /users/PAS3560/clute/data data
```

The shared datasets are readable by all OSC users. Dataset preparation keeps
files readable and directories traversable without adding write access.
Each selected dataset must already have a version
directory, such as `data/v001.visdrone/`, and its prepared archive at
`data/archives/v001.visdrone.tar`.

### Import a Roboflow export

Select the YOLO detection format in Roboflow and copy the download command.
From the repository root, run `python prepare_datasets.py -rf` and paste the full
command at the hidden prompt. You can also wrap the full command in single
quotes when passing it directly, or supply a ZIP you already downloaded.

```bash
python prepare_datasets.py -rf 'curl -L "https://app.roboflow.com/ds/EXPORT?key=YOUR_KEY" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip'
python prepare_datasets.py --zip ../roboflow.zip
```

The script submits an OSC CPU job that creates the next numbered version under
`data/`, with `yolo/`, `coco/`, and a tar under `data/archives/`. Wait for the
job to finish before training on the latest version. Use
`python prepare_datasets.py --existing` to rebuild archives for existing datasets.

### 4. Log in to W&B

```bash
wandb login
```

Use your own W&B API key for an account with access to `BuckeyeVertical`.
Each framework directory defines its W&B behavior in `__wandb__.py`. When
training starts, `slurmable` loads that file beside the model script and runs
`train(run)` inside its `track(run)` context. Importing a model or submitting a
job does not load the adapter or connect to W&B.

The Ultralytics adapter registers logging callbacks; the LightlyTrain and
RF-DETR adapters supply native logger settings. The adapters handle worker ranks
and close W&B on success or failure. Model files contain only training code and settings.
Runs use `BuckeyeVertical/season-2026-2027`. To add another framework, place a
`__wandb__.py` exposing a `track(run)` context manager beside its model scripts.
Directories without an adapter run training directly.

All adapters read the entity from `run.wandb_entity`, which defaults to
`BuckeyeVertical` in `base.Run`. LightlyTrain and RF-DETR pass this value through
a temporary `WANDB_ENTITY` setting and restore the previous environment after
training. RF-DETR uses its native Lightning W&B logger, with `WANDB_RUN_ID` and
`WANDB_RUN_GROUP` supplying the reserved run ID and model group.

## Submit a training job

Once setup is complete and the shared dataset is prepared, run a model file:

```bash
python models/ultralytics/yolo26s.py
# Or:
python models/lightly/ltdetrv2-s.py
# Or:
python models/rfdetr/rfdetr-small.py
# Or:
python models/transformers/rtdetr-r18.py
```

For MMDetection, activate `bv_ml_mmdetection` and run:

```bash
python models/mmdetection/faster-rcnn-r50-fpn.py
```

This uses the native Faster R-CNN ResNet-50 FPN configuration: 12 epochs,
batch size 2, and automatic learning-rate scaling on one GPU. Its data adapter
supplies COCO datasets, evaluators, class names, and data-loader workers. The
model sets its output directory and detection-head class count; W&B uses
MMEngine's native `WandbVisBackend`. Backbone weights are cached in
`.modelcache/hub`. As with the other frameworks, `DATA_VERSION` selects the
dataset, and the test split is optional.

The Transformers entry point fine-tunes [RT-DETR ResNet-18](https://huggingface.co/PekingU/rtdetr_r18vd)
through Hugging Face's native `Trainer` in the main `bv_ml` environment. Its
three-file framework directory follows the same data and W&B adapter convention.
The data adapter reads prepared COCO train/validation splits, preserves nested
image paths, maps category IDs to contiguous model labels, and supplies image
processing and COCO mAP metrics. The model requests one GPU and 24 CPUs for four
hours, with 50 epochs and training batch size 4. Evaluation processes one image
per batch to preserve variable-length COCO annotations in the native Trainer.
Checkpoints go into `run.train_out_dir`;
pretrained weights and processor files are cached in `.modelcache/huggingface`.
The W&B adapter configures the native Trainer integration using the reserved run
ID, season project, model group, and configured entity. `DATA_VERSION` and staged
datasets work as usual. See the [Transformers detection guide](https://huggingface.co/docs/transformers/tasks/object_detection)
for its training API.

The file submits its job through `sbatch` automatically. The current model uses
Pitzer's `gpu-exp` partition under `PAS3560`. Default Slurm resources live in
`base.py`; model decorators override them. Training settings live in each model.

Model entry points use `train(run)`. The `run` object provides `data_dir` (the
selected dataset, staged on the compute node), `data` (the framework's dataset
configuration), `train_out_dir`, `out_dir`,
`model_cache_dir`, and Slurm resources. Set `DATA_VERSION=1` in `@slurmable(...)`
to pin a dataset; the default, `-1`, selects the latest version at submission.

Each framework directory defines its data connection in `__data__.py` with a
`load(run)` function. The launcher calls it after resolving the staged dataset
path and before entering W&B tracking, then supplies its return value as
`run.data`. Model scripts pass this configuration to their framework. Ultralytics
receives the YOLO YAML path; LightlyTrain receives a YOLO configuration mapping
with the dataset root and class names resolved by its adapter. RF-DETR receives
a COCO dataset view as `dataset_dir=run.data`. Its adapter uses the prepared
`coco/annotations/instances_{train,val,test}.json` files, renames `val` to
`valid`, and links the staged images under `run.out_dir/data`. Only annotation
metadata is copied; the shared dataset stays unchanged and nested image paths
are preserved. Importing a model or submitting a job does not load the data
adapter. Directories without one
keep `run.data=None` and can use `run.data_dir` directly.

The RF-DETR Small entry point requests one GPU and 24 CPUs for four hours,
trains for 100 epochs with batch size 4 and four accumulation steps, and caches
its pretrained checkpoint in `.modelcache/rf-detr-small.pth`. Checkpoints and
native training logs go in `run.train_out_dir`, alongside the existing Slurm
logs under the reserved run directory. Prepare the COCO annotations and archive
with `python prepare_datasets.py` before training; both `train` and `val`
splits are required, while the test split is optional. See the official
[dataset format](https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/)
and [training parameter](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/)
references.

To train inside an existing allocation, append `--train` to the model command.

The launcher selects the highest numeric dataset version and extracts its
archive into the compute node's `$TMPDIR` before training. It writes results
and Slurm logs under `out/2026-2027/` in your checkout. W&B runs use the
`BuckeyeVertical/season-2026-2027` project.

For later sessions, activate `bv_ml` and run the model file from the repository
root. Environment creation, the dataset symlink, and W&B login are setup steps.
