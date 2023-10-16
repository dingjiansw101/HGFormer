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
_RAW_ACDC_SPLITS = {
    "city_c_gaussiannoise5_vis": ("gauss_noise/5/", "cityscapes/gtFine/val/"),
    "city_c_gaussiannoise4_vis": ("gauss_noise/4/", "cityscapes/gtFine/val/"),
    "city_c_gaussiannoise3_vis": ("gauss_noise/3/", "cityscapes/gtFine/val/"),
    "city_c_gaussiannoise2_vis": ("gauss_noise/2/", "cityscapes/gtFine/val/"),
    "city_c_gaussiannoise1_vis": ("gauss_noise/1/", "cityscapes/gtFine/val/"),
    "city_c_gaussiannoise0_vis": ("gauss_noise/0/", "cityscapes/gtFine/val/"),
    "city_c_tmp_gaussiannoise4_vis": ("city_c_tmp/gaussian_noise/4/", "cityscapes/gtFine/val/"),
    "city_c_tmp_clean_vis": ("city_c_tmp/clean/", "cityscapes/gtFine/val/"),

}

def register_all_city_c_vis(root):
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
register_all_city_c_vis(_root)
