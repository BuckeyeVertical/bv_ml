from base import Run, slurmable


@slurmable(gpus_per_node=1, cpus_per_task=24, time="04:00:00")
def train(run: Run):
    import torch
    from mmengine.hub import get_config
    from mmengine.runner import Runner

    if run.slurm["gpus-per-node"] != 1:
        raise ValueError("This MMDetection entry point requires gpus_per_node=1")

    cfg = get_config("mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py")
    cfg.merge_from_dict(run.data)
    cfg.work_dir = str(run.train_out_dir)
    cfg.auto_scale_lr.enable = True
    cfg.model.roi_head.bbox_head.num_classes = len(cfg.metainfo.classes)
    torch.hub.set_dir(str(run.model_cache_dir / "hub"))
    Runner.from_cfg(cfg).train()
