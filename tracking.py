from functools import wraps

import base


def wandb_run(model, train_args=None):
    import wandb

    output_dir = model.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    group = model.model_path.as_posix()
    run_config = {
        "model_path": group,
        "model": model.model_path.name,
        "run_number": model._run_number,
        "run_timestamp": model._run_timestamp,
        "data_version": model._resolved_data_version,
    }
    if train_args is not None:
        run_config["train_args"] = train_args
    return wandb.init(
        project=base.ROOT.name,
        mode="online",
        group=group,
        name=f"{group}/{output_dir.name}",
        dir=str(output_dir),
        config=run_config,
    )


def tracked_train(train):
    """Start one online W&B run around a model's training method."""
    @wraps(train)
    def wrapped(self, *args, **kwargs):
        # An override may call super().train(); keep that in the same run.
        if getattr(self, "_wandb_training", False):
            return train(self, *args, **kwargs)
        with wandb_run(self):
            self._wandb_training = True
            try:
                return train(self, *args, **kwargs)
            finally:
                self._wandb_training = False

    return wrapped
