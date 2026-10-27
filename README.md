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

### 4. Log in to W&B

```bash
wandb login
```

Use your own W&B API key for an account with access to `BuckeyeVertical`.
Each framework directory defines its W&B behavior in `__wandb__.py`. When
training starts, `slurmable` loads that file beside the model script and runs
`train(run)` inside its `track(run)` context. Importing a model or submitting a
job does not load the adapter or connect to W&B.

The Ultralytics adapter registers logging callbacks; the LightlyTrain adapter
supplies native logger settings. Both handle worker ranks and close W&B on
success or failure. Model files contain only training code and settings.
Runs use `BuckeyeVertical/season-2026-2027`. To add another framework, place a
`__wandb__.py` exposing a `track(run)` context manager beside its model scripts.
Directories without an adapter run training directly.

Both adapters read the entity from `run.wandb_entity`, which defaults to
`BuckeyeVertical` in `base.Run`. LightlyTrain passes this value through a temporary
`WANDB_ENTITY` setting and restores the previous environment after training.

## Submit a training job

Once setup is complete and the shared dataset is prepared, run a model file:

```bash
python models/ultralytics/yolo26s.py
# Or:
python models/lightly/ltdetrv2-s.py
```

The file submits its job through `sbatch` automatically. The current model uses
Pitzer's `gpu-exp` partition under `PAS3560`. Default Slurm resources live in
`base.py`; model decorators override them. Training settings live in each model.

Model entry points use `train(run)`. The `run` object provides `data_dir` (the
selected dataset, staged on the compute node), `train_out_dir`, `out_dir`,
`model_cache_dir`, and Slurm resources. Set `DATA_VERSION=1` in `@slurmable(...)`
to pin a dataset; the default, `-1`, selects the latest version at submission.

To train inside an existing allocation, append `--train` to the model command.

The launcher selects the highest numeric dataset version and extracts its
archive into the compute node's `$TMPDIR` before training. It writes results
and Slurm logs under `out/2026-2027/` in your checkout. W&B runs use the
`BuckeyeVertical/season-2026-2027` project.

For later sessions, activate `bv_ml` and run the model file from the repository
root. Environment creation, the dataset symlink, and W&B login are setup steps.
