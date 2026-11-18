from base import Run, slurmable


@slurmable(cpus_per_task=24, time="04:00:00")
def train(run: Run):
    from rfdetr import RFDETRSmall

    RFDETRSmall(pretrain_weights=str(run.model_cache_dir / "rf-detr-small.pth")).train(
        dataset_dir=run.data,
        output_dir=str(run.train_out_dir),
        devices=run.slurm["gpus-per-node"],
        num_workers=run.slurm["cpus-per-task"],
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
    )
