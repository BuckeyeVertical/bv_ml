"""W&B setup for every LightlyTrain model in this directory."""

import os
from contextlib import contextmanager
from functools import wraps

from base import Run


@contextmanager
def track(run: Run):
    import lightly_train
    import wandb

    original_train = lightly_train.train_object_detection

    @wraps(original_train)
    def train(*args, logger_args=None, **kwargs):
        loggers = dict(logger_args or {})
        loggers["wandb"] = {
            "project": f"season-{run.season}",
            "name": run.out_dir.name,
            "version": run.run_id,
        }
        return original_train(*args, logger_args=loggers, **kwargs)

    environment = {
        "WANDB_ENTITY": run.wandb_entity,
        "WANDB_RUN_GROUP": run.model_path.as_posix(),
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    lightly_train.train_object_detection = train
    exit_code = 1
    try:
        yield
        exit_code = 0
    finally:
        lightly_train.train_object_detection = original_train
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        wandb.finish(exit_code=exit_code)
