#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from multiprocessing import Pool

id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}


def convert(input, outputpath):
    lab = np.asarray(Image.open(input))
    assert lab.dtype == np.uint8
    output = np.zeros_like(lab, dtype=np.uint8) + 255
    for obj_id in np.unique(lab):
        if obj_id in id_to_trainid:
            output[lab == obj_id] = id_to_trainid[obj_id]

    Image.fromarray(output).save(outputpath)

def worker(file_tuple):
    file, output_file = file_tuple
    lab = np.asarray(Image.open(file))
    assert lab.dtype == np.uint8
    output = np.zeros_like(lab, dtype=np.uint8) + 255
    for obj_id in np.unique(lab):
        if obj_id in id_to_trainid:
            output[lab == obj_id] = id_to_trainid[obj_id]

    Image.fromarray(output).save(output_file)

if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "GTA"
    for name in ["train", "valid", "test"]:
        annotation_dir = dataset_dir / "labels" / name
        output_dir = dataset_dir / "labels_detectron2" / name
        output_dir.mkdir(parents=True, exist_ok=True)

        file_list = []
        for file in tqdm.tqdm(list(annotation_dir.iterdir())):
            output_file = output_dir / file.name
            file_list.append((file, output_file))
            # convert(file, output_file)

        pool = Pool(32)
        pool.map(worker, file_list)
        print(f'done {name}')
