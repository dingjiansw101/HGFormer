#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from multiprocessing import Pool

ignore_label = 255

id_to_ignore_or_group = {}

# def gen_id_to_ignore():
    # global id_to_ignore_or_group
for i in range(66):
    id_to_ignore_or_group[i] = ignore_label

### Convert each class to a corresponding cityscapes class
### Road
# Road
id_to_ignore_or_group[13] = 0
# Lane Marking - General
id_to_ignore_or_group[24] = 0
# Manhole
id_to_ignore_or_group[41] = 0

### Sidewalk
# Curb
id_to_ignore_or_group[2] = 1
# Sidewalk
id_to_ignore_or_group[15] = 1

### Building
# Building
id_to_ignore_or_group[17] = 2

### Wall
# Wall
id_to_ignore_or_group[6] = 3

### Fence
# Fence
id_to_ignore_or_group[3] = 4

### Pole
# Pole
id_to_ignore_or_group[45] = 5
# Utility Pole
id_to_ignore_or_group[47] = 5

### Traffic Light
# Traffic Light
id_to_ignore_or_group[48] = 6

### Traffic Sign
# Traffic Sign
id_to_ignore_or_group[50] = 7

### Vegetation
# Vegitation
id_to_ignore_or_group[30] = 8

### Terrain
# Terrain
id_to_ignore_or_group[29] = 9

### Sky
# Sky
id_to_ignore_or_group[27] = 10

### Person
# Person
id_to_ignore_or_group[19] = 11

### Rider
# Bicyclist
id_to_ignore_or_group[20] = 12
# Motorcyclist
id_to_ignore_or_group[21] = 12
# Other Rider
id_to_ignore_or_group[22] = 12

### Car
# Car
id_to_ignore_or_group[55] = 13

### Truck
# Truck
id_to_ignore_or_group[61] = 14

### Bus
# Bus
id_to_ignore_or_group[54] = 15

### Train
# On Rails
id_to_ignore_or_group[58] = 16

### Motorcycle
# Motorcycle
id_to_ignore_or_group[57] = 17

### Bicycle
# Bicycle
id_to_ignore_or_group[52] = 18



def convert(filetuple):
    input, outputpath = filetuple
    lab = np.asarray(Image.open(input))
    assert lab.dtype == np.uint8
    output = np.zeros_like(lab, dtype=np.uint8) + 255
    for obj_id in np.unique(lab):
        # print(f'obj_id{obj_id}')
        # print(f'{id_to_ignore_or_group}')
        if obj_id in id_to_ignore_or_group:
            output[lab == obj_id] = id_to_ignore_or_group[obj_id]

    Image.fromarray(output).save(outputpath)

if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "mapillary"
    pool = Pool(32)
    # gen_id_to_ignore()
    # import ipdb; ipdb.set_trace()
    for name in ["training", "validation"]:
        annotation_dir = dataset_dir / name / "labels"
        output_dir = dataset_dir / "labels_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        filelist = []
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            # convert(file, output_file)
            filelist.append((file, output_file))
        pool.map(convert, filelist)