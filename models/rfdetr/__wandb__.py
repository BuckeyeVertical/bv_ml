"""W&B setup for every RF-DETR model in this directory."""

import os
from contextlib import contextmanager
from functools import wraps

from base import Run


@contextmanager
def track(run: Run):
    import wandb
    from rfdetr import RFDETR

    original_train = RFDETR.train

    @wraps(original_train)
    def train(model, **kwargs):
        kwargs.update(wandb=True, project=f"season-{run.season}", run=run.out_dir.name)
        return original_train(model, **kwargs)

    environment = {
        "WANDB_ENTITY": run.wandb_entity,
        "WANDB_RUN_GROUP": run.model_path.as_posix(),
        "WANDB_RUN_ID": run.run_id,
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    RFDETR.train = train
    exit_code = 1
    try:
        yield
        exit_code = 0
    finally:
        RFDETR.train = original_train
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        wandb.finish(exit_code=exit_code)
