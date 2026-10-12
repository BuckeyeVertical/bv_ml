"""Small W&B connection shared by framework classes."""

import os


def wandb_run(config, train_args=None):
    """Return a W&B run context named after the local output directory."""
    import wandb

    output_dir = config.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    group = f"{config.framework}/{config.job_name}"
    return wandb.init(
        project=config.base_dir.name,
        mode=os.environ.get("WANDB_MODE", "disabled"),
        group=group,
        name=f"{group}/{output_dir.name}",
        dir=str(output_dir),
        config={
            "framework": config.framework,
            "model": config.job_name,
            "run_version": config._run,
            "train_args": train_args,
        },
    )
