from base import slurmable


@slurmable(
    DATA_VERSION=1,
    clusters="ascend",
    partition="next-gen",
    gpus_per_node=1,
    cpus_per_task=16,
    time="02:00:00",
)
def train(config):
    from ultralytics import YOLO

    return YOLO(str(config.model_cache_dir / "yolo26l.pt")).train(
        data=str(config.data_dir / "yolo" / "data.yaml"),
        save_dir=config.out_dir,
        epochs=100,
        imgsz=640,
        batch=4,
        amp=True,
        cache=False,
        workers=config.slurm["cpus-per-task"] // 4,
    )
