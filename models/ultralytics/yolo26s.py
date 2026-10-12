from base import Pitzer, configure, slurmable
from models.ultralytics.framework import UltralyticsModel


@slurmable
@configure(partition=Pitzer.gpu_exp, gpus_per_node=1)
class yolo26s(UltralyticsModel):
    def get_model(self):
        from ultralytics import YOLO
        return YOLO("yolo26s.pt")

    def get_overrides(self):
        return {}
