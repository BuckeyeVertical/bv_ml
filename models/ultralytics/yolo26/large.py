from base import BaseModel, configure, slurmable


@slurmable
@configure(slurm={
    "clusters": "pitzer",
    "partition": "gpu-exp",
    "gpus-per-node": 1,
    "cpus-per-task": 16,
    "time": "2-00:00:00",
})
class Large(BaseModel):
    def train(self):
        from ultralytics import YOLO

        return YOLO(str(self.model_cache_dir / "yolo26l.pt")).train(
            data=str(self.data_dir / "yolo" / "data.yaml"),
            save_dir=self.out_dir,
            epochs=100,
            imgsz=640,
            batch=4,
            amp=True,
            cache=False,
            workers=self.config.slurm["cpus-per-task"] // 4,
        )
