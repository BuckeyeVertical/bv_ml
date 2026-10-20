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

In your new checkout, create `data/` as a symlink to Trevor's dataset directory:

```bash
ln -s /fs/scratch/PAS3560/trevor/bv_ml/data data
```

This points to the shared storage at `/fs/scratch/PAS3560/data`. Training only
needs read access to it. Each selected dataset must already have a version
directory, such as `data/v001.visdrone/`, and its prepared archive at
`data/archives/v001.visdrone.tar`.

### 4. Log in to W&B

```bash
wandb login
```

Use your own W&B API key for an account with access to `BuckeyeVertical`.
The training wrapper starts a W&B run using the settings in `base.py`.

## Submit a training job

Once setup is complete and the shared dataset is prepared, run a model file:

```bash
python models/ultralytics/yolo26s.py
```

The file submits its job through `sbatch` automatically. The current model uses
Pitzer's `gpu-exp` partition under `PAS3560`; its Slurm resources and training
settings are defined in the model file.

The launcher selects the highest numeric dataset version and extracts its
archive into the compute node's `$TMPDIR` before training. It writes results
and Slurm logs under `out/2026-2027/` in your checkout. W&B runs use the
`BuckeyeVertical/season-2026-2027` project.

For later sessions, activate `bv_ml` and run the model file from the repository
root. Environment creation, the dataset symlink, and W&B login are setup steps.

## Prepare datasets when needed

Skip this step when the shared dataset versions and their archives are already
prepared. A maintainer with write access to the dataset directory can submit
preparation for new or unprepared versions:

```bash
python prepare_datasets.py
```

This submits a CPU job to Pitzer's `cpu-exp` partition. It normalizes dataset
names, imports ZIP files, converts YOLO annotations to COCO, and creates the
`.tar` archives needed for training. Failed versions are skipped.

Wait for preparation to finish and confirm the selected version succeeded
before submitting training. Logs and JSON summaries are saved under
`out/dataset-preparation/` in the checkout that submitted preparation.
