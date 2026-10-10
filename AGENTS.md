# Python environment

Use the existing Conda environment named `bv_ml` for all Python execution,
validation, training, and dependency installation in this repository.

Prefer explicit commands so the environment is selected consistently:

```bash
conda run -n bv_ml --no-capture-output python <script.py>
conda run -n bv_ml --no-capture-output python -m unittest discover
conda run -n bv_ml --no-capture-output python -m pip install -r requirements.txt
```

An interactive shell may instead use `conda activate bv_ml` before running
Python commands. Submit Slurm jobs from `bv_ml`; submission uses
`sys.executable`, so the compute-node command uses that environment's Python.
