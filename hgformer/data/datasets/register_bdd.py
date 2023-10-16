# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata


_RAW_BDD_SPLITS = {
    "bdd_train": ("bdd/images/10k/train", "bdd/labels/sem_seg/masks/train"),
    "bdd_val": ("bdd/images/10k/val", "bdd/labels/sem_seg/masks/val")
}

def register_all_bdd(root):
    for key, (image_dir, gt_dir) in _RAW_BDD_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        # DatasetCatalog.register(
        #     key, lambda x=image_dir, y=gt_dir: load_sem_seg(x, y)
        # )
        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_bdd(_root)