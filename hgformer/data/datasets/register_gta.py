# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

# ==== Predefined splits for raw gta images ===========

GTA_Trainid = {
    "gta_trainid_val": ("gta/images/valid/", "gta/labels_detectron2/valid/"),
}

def register_all_gta_sem_seg(root):
    for key, (image_dir, gt_dir) in GTA_Trainid.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")

register_all_gta_sem_seg(_root)