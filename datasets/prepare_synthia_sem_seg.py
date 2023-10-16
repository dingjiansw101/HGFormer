#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from multiprocessing import Pool
import cv2
import imageio
import imageio.v2 as imageio
ignore_label = 255

# mapping based on README.txt from SYNTHIA_RAND_CITYSCAPES
trainid_to_trainid = {
        0: ignore_label,  # void
        1: 10,            # sky
        2: 2,             # building
        3: 0,             # road
        4: 1,             # sidewalk
        5: 4,             # fence
        6: 8,             # vegetation
        7: 5,             # pole
        8: 13,            # car
        9: 7,             # traffic sign
        10: 11,           # pedestrian - person
        11: 18,           # bicycle
        12: 17,           # motorcycle
        13: ignore_label, # parking-slot
        14: ignore_label, # road-work
        15: 6,            # traffic light
        16: 9,            # terrain
        17: 12,           # rider
        18: 14,           # truck
        19: 15,           # bus
        20: 16,           # train
        21: 3,            # wall
        22: ignore_label  # Lanemarking
        }

# def convert(filetupe):
#     input, outputpath = filetupe
#     # lab = np.asarray(Image.open(input))
#     # lab = imageio.imread(input, format='PNG-FI')
#     lab = imageio.imread(input, format='PNG')
#
#     # print(input)
#     # lab = cv2.imread(str(input), cv2.IMREAD_UNCHANGED)[:, :, -1]
#     lab = np.array(lab, dtype=np.uint8)[:, :, 0]
#     assert lab.dtype == np.uint8
#     output = np.zeros_like(lab, dtype=np.uint8) + 255
#     for obj_id in np.unique(lab):
#         if obj_id in trainid_to_trainid:
#             output[lab == obj_id] = trainid_to_trainid[obj_id]
#
#     Image.fromarray(output).save(outputpath)


def convert(filetupe):
    file, new_file = filetupe
    # re-assign labels to match the format of Cityscapes
    # PIL does not work with the image format, but cv2 does
    label = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)[:, :, -1]

    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in trainid_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    # new_file = file.replace('.png', '_labelTrainIds.png')
    # assert file != new_file
    # sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    # return sample_class_stats

if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "synthia"
    pool = Pool(32)
    for name in ["train", "val"]:
    # for name in ["train"]:
        annotation_dir = dataset_dir / "GT" / "LABELS" / name
        output_dir = dataset_dir / "labels_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        filelist = []
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            # convert(file, output_file)
            filelist.append((file, output_file))
        pool.map(convert, filelist)