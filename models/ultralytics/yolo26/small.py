from base import BaseModel, configure, slurmable


@slurmable
@configure(slurm={
    "clusters": "pitzer",
    "partition": "gpu-exp",
    "gpus-per-node": 1,
    "cpus-per-task": 8,
    "time": "01:00:00",
})
class Small(BaseModel):
    def train(self):
        from ultralytics import YOLO

        YOLO(str(self.model_cache_dir / "yolo26s.pt")).train(
            data=str(self.data_dir / "yolo" / "data.yaml"),
            save_dir=self.out_dir,
            epochs=100,
            imgsz=640,
            batch=16,
            amp=True,
            cache=False,
            workers=self.config.slurm["cpus-per-task"] // 4,
        )
