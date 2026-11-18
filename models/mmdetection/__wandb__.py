"""W&B setup for every MMDetection model in this directory."""

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps

from base import Run


@contextmanager
def track(run: Run):
    import wandb
    from mmengine.runner import Runner

    original_build_visualizer = Runner.build_visualizer

    @wraps(original_build_visualizer)
    def build_visualizer(runner, visualizer=None):
        visualizer = deepcopy(visualizer or {
            "type": "DetLocalVisualizer", "vis_backends": [{"type": "LocalVisBackend"}],
        })
        backends = [
            backend for backend in visualizer.get("vis_backends", [])
            if backend.get("type") != "WandbVisBackend"
        ]
        backends.append({
            "type": "WandbVisBackend",
            "init_kwargs": {
                "entity": run.wandb_entity,
                "project": f"season-{run.season}",
                "group": run.model_path.as_posix(),
                "name": run.out_dir.name,
                "id": run.run_id,
                "dir": str(run.out_dir),
            },
        })
        visualizer["vis_backends"] = backends
        return original_build_visualizer(runner, visualizer)

    Runner.build_visualizer = build_visualizer
    exit_code = 1
    try:
        yield
        exit_code = 0
    finally:
        Runner.build_visualizer = original_build_visualizer
        wandb.finish(exit_code=exit_code)
