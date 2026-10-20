import contextlib
import io
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import base


class ModelOutputTests(unittest.TestCase):
    def test_ddp_startup_preserves_slurm_logs_and_tracking_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.dict(os.environ, {"YOLO_CONFIG_DIR": str(root / "yolo-config")}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                from ultralytics.utils import dist

            config = base.Run(
                data_dir=root / "data",
                model_cache_dir=root / "cache",
                out_dir=root / "yolo26s",
                slurm={"cpus-per-task": 24},
                model_path=Path("ultralytics/yolo26s"),
                run_id="0123456789abcdef0123456789abcdef",
                run_timestamp="20261020T142813000000Z",
                data_version=1,
            )
            files = [config.out_dir / "logs/slurm-123.out", config.out_dir / "wandb/run.log"]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("active run\n")

            def prepare_ddp(**options):
                save_dir = Path(options["save_dir"])
                (save_dir / "weights").mkdir(parents=True, exist_ok=True)
                trainer = SimpleNamespace(save_dir=save_dir, resume=False, world_size=2)
                dist.generate_ddp_command(trainer)

            model = runpy.run_path(str(base.ROOT / "models/ultralytics/yolo26s.py"))
            with (
                patch("ultralytics.YOLO") as yolo,
                patch.object(dist, "generate_ddp_file", return_value="unused-ddp.py"),
                patch.object(dist, "find_free_network_port", return_value=29500),
            ):
                yolo.return_value.train.side_effect = prepare_ddp
                model["train"].__wrapped__(config)
                yolo.assert_called_once_with(str(config.model_cache_dir / "yolo26s.pt"))
                yolo.return_value.train.assert_called_once()

            for path in files:
                self.assertTrue(path.is_file(), f"DDP startup deleted {path}")
                self.assertEqual(path.read_text(), "active run\n")


if __name__ == "__main__":
    unittest.main()
