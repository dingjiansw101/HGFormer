# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
# from .acdc import load_acdc_semantic
from detectron2.data.datasets.cityscapes import load_cityscapes_semantic

corruptions = ['clean', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
# ==== Predefined splits for raw cityscapes c images ===========

_RAW_ACDC_SPLITS = {}
for noise in corruptions:
    if noise == 'clean':
        cur_data = {f"cityscapes_fine_{noise}_val": (f"cityscapes-c/{noise}/", "cityscapes/gtFine/val/")}
    else:
        for severity in range(5):
            severity_str = str(severity+1)
            cur_data = {f"cityscapes_fine_{noise}_{severity_str}_val": (f"cityscapes-c/{noise}/{severity_str}", "cityscapes/gtFine/val/")}
            _RAW_ACDC_SPLITS.update(cur_data)
def register_all_city_c(root):
    for key, (image_dir, gt_dir) in _RAW_ACDC_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        # sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_city_c(_root)
