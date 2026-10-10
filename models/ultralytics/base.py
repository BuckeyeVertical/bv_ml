from base import SlurmModel


class UltralyticsModel(SlurmModel):
    config = SlurmModel.config.with_options(framework="ultralytics")

    def get_model(self):
        from ultralytics import YOLO
        return YOLO("yolo26n.pt")

    def get_overrides(self):
        return {}

    def train(self):
        config = self.config
        output_dir = config.out_dir
        train_args = {
            "data": config.data,
            "save_dir": str(output_dir),
            **self.get_overrides(),
        }
        print(train_args)
        model = self.get_model()
        return model.train(**train_args)
