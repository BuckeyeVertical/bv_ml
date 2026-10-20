from base import slurmable


@slurmable(
    clusters="pitzer",
    partition="gpu-exp",
    gpus_per_node=2,
    cpus_per_task=24,
    time="01:00:00",
)
def train(config):
    from ultralytics import YOLO

    YOLO(str(config.model_cache_dir / "yolo26s.pt")).train(
        data=str(config.data_dir / "yolo" / "data.yaml"),
        save_dir=config.train_out_dir,
        epochs=100,
        imgsz=640,
        batch=4,
    )
