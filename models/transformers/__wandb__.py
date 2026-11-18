"""W&B setup for every Transformers model in this directory."""

import os
from contextlib import contextmanager
from functools import wraps

from base import Run


@contextmanager
def track(run: Run):
    import wandb
    from transformers import Trainer

    original_init = Trainer.__init__

    @wraps(original_init)
    def initialize(trainer, *args, **kwargs):
        kwargs["args"].report_to = ["wandb"]
        kwargs["args"].run_name = run.out_dir.name
        return original_init(trainer, *args, **kwargs)

    environment = {
        "WANDB_ENTITY": run.wandb_entity,
        "WANDB_PROJECT": f"season-{run.season}",
        "WANDB_RUN_GROUP": run.model_path.as_posix(),
        "WANDB_RUN_ID": run.run_id,
        "WANDB_DIR": str(run.out_dir),
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    Trainer.__init__ = initialize
    exit_code = 1
    try:
        yield
        exit_code = 0
    finally:
        Trainer.__init__ = original_init
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        wandb.finish(exit_code=exit_code)
