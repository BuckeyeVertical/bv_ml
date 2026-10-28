from base import Run, slurmable


@slurmable(cpus_per_task=24, time="04:00:00")
def train(run: Run):
    import lightly_train

    lightly_train.train_object_detection(
        out=str(run.train_out_dir),
        model="ltdetrv2-s-coco",
        steps=5_000,
        batch_size=16,
        gradient_accumulation_steps=2,
        precision="16-mixed",
        devices=run.slurm["gpus-per-node"],
        model_args={
            "lr": 2e-4,
            "scheduler_name": "flat-cosine",
            "lr_warmup_steps": 200,
            "scheduler_flat_steps": 1_500,
        },
        metric_args={"classwise": True},
        data=run.data,
    )
