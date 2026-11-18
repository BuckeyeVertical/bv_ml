from base import Run, slurmable


@slurmable(gpus_per_node=1, cpus_per_task=24, time="04:00:00")
def train(run: Run):
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        Trainer,
        TrainingArguments,
    )

    if run.slurm["gpus-per-node"] != 1:
        raise ValueError("This Transformers entry point requires gpus_per_node=1")

    checkpoint = "PekingU/rtdetr_r18vd"
    cache = str(run.model_cache_dir / "huggingface")
    processor = AutoImageProcessor.from_pretrained(checkpoint, cache_dir=cache)
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint, cache_dir=cache, ignore_mismatched_sizes=True, **run.data["labels"],
    )
    Trainer(
        model=model, **run.data["trainer"](processor),
        args=TrainingArguments(
            output_dir=str(run.train_out_dir), num_train_epochs=50, learning_rate=1e-4,
            # Single-image evaluation preserves ragged COCO labels in Trainer's gathering.
            per_device_train_batch_size=4, per_device_eval_batch_size=1,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            **run.data["loader_args"],
        ),
    ).train()
