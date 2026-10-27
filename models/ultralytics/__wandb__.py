"""W&B setup for every Ultralytics model in this directory."""

from contextlib import contextmanager
from importlib import reload

from base import Run


@contextmanager
def track(run: Run):
    import wandb
    from ultralytics import settings
    from ultralytics.utils import callbacks

    settings.update({"wandb": True})
    from ultralytics.utils.callbacks import wb

    # An earlier training call may have imported the integration while disabled.
    if wb.wb is None:
        reload(wb)

    def start_wandb(trainer):
        wandb.init(
            entity=run.wandb_entity,
            project=f"season-{run.season}",
            group=run.model_path.as_posix(),
            name=run.out_dir.name,
            id=run.run_id,
            dir=str(run.out_dir),
            config=vars(trainer.args),
        )

    # The primary training worker runs this callback, including after DDP launch.
    startup = callbacks.default_callbacks["on_pretrain_routine_start"]
    startup.insert(0, start_wandb)
    exit_code = 1
    try:
        yield
        exit_code = 0
    finally:
        startup.remove(start_wandb)
        wandb.finish(exit_code=exit_code)
