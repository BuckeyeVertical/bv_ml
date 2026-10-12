import os
import unittest
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from models.ultralytics.framework import UltralyticsModel


class FrameworkTrackingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.wb = SimpleNamespace(init=MagicMock())
        self.model = SimpleNamespace(train=Mock(return_value="trained"))
        model = self.model

        class Framework(UltralyticsModel):
            config = replace(UltralyticsModel.config, job_name="yolo26s", data_version="v1", _run=2)

            def get_model(self):
                return model

        self.framework = Framework
        for patcher in (
            patch("base.ROOT", Path(temporary.name)),
            patch.dict("sys.modules", {"wandb": self.wb}),
            patch.dict(os.environ),
            patch("sys.stdout", new_callable=StringIO),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("WANDB_MODE", None)

    def test_framework_connects_with_output_names_before_training(self):
        def train(**kwargs):
            self.wb.init.assert_called_once()
            self.wb.init.return_value.__enter__.assert_called_once()
            return "trained"

        self.model.train.side_effect = train
        self.assertEqual(self.framework().train(), "trained")
        kwargs = self.wb.init.call_args.kwargs
        self.assertEqual(kwargs["project"], self.framework.config.base_dir.name)
        self.assertEqual(kwargs["group"], "ultralytics/yolo26s")
        self.assertEqual(kwargs["name"], "ultralytics/yolo26s/v2")
        self.assertEqual(kwargs["config"]["train_args"], self.model.train.call_args.kwargs)

    def test_cloud_sync_requires_explicit_environment_mode(self):
        self.framework().train()
        self.assertEqual(self.wb.init.call_args.kwargs["mode"], "disabled")
        os.environ["WANDB_MODE"] = "online"
        self.framework().train()
        self.assertEqual(self.wb.init.call_args.kwargs["mode"], "online")

    def test_framework_respects_offline_environment(self):
        os.environ["WANDB_MODE"] = "offline"
        self.framework().train()
        self.assertEqual(self.wb.init.call_args.kwargs["mode"], "offline")

    def test_children_inherit_framework_connection(self):
        class Inherited(self.framework):
            pass

        Inherited().train()
        self.wb.init.assert_called_once()
        self.model.train.assert_called_once()

    def test_training_exception_reaches_run_context(self):
        error = ValueError("training failed")
        self.model.train.side_effect = error
        with self.assertRaisesRegex(ValueError, "training failed"):
            self.framework().train()
        arguments = self.wb.init.return_value.__exit__.call_args.args
        self.assertIs(arguments[0], ValueError)
        self.assertIs(arguments[1], error)


if __name__ == "__main__":
    unittest.main()
