from base import Run, slurmable


@slurmable(gpus_per_node=2, cpus_per_task=24)
def train(run: Run):
    from ultralytics import YOLO

    YOLO(str(run.model_cache_dir / "yolo26s.pt")).train(
        data=str(run.data_dir / "yolo" / "data.yaml"),
        save_dir=run.train_out_dir,
        device=list(range(run.slurm["gpus-per-node"])),
        epochs=100,
        imgsz=640,
        batch=4,
    )
