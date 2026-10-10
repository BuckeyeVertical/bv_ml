from base import Pitzer, slurmable
from models.ultralytics.base import UltralyticsModel


@slurmable
class yolo26s(UltralyticsModel):
    config = UltralyticsModel.config.with_options(
        partition=Pitzer.gpu_exp,
        gpus_per_node=1,
    )

    def get_model(self):
        from ultralytics import YOLO 
        return YOLO("yolo26s.pt")

    def get_overrides(self):
        return {}
