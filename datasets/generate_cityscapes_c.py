from imagecorruptions import corrupt
from imagecorruptions import get_corruption_names
import os
import cv2
from multiprocessing import Pool
import numpy as np
import random
import mmcv

random.seed(8) # for reproducibility
np.random.seed(8)
corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']

img_dir = 'datasets/cityscapes-c/clean'
num_imgs = 500
img_names = []
prog_bar = mmcv.ProgressBar(num_imgs)
img_dict = {}
for img_path in mmcv.scandir(img_dir, suffix='png', recursive=True):
    img_name = os.path.join(img_dir, img_path)
    img = mmcv.imread(img_name)
    img_dict[img_name] = img
    prog_bar.update()

def perturb(i, p, s):
    img = corrupt(i, corruption_name=p, severity=s)
    return img

def worker(optuple):
    srcfile, p, s, perturbed_img_path = optuple
    img = img_dict[srcfile]
    perturbed_img = perturb(img, p, s)
    mmcv.imwrite(perturbed_img, perturbed_img_path, auto_mkdir=True)

def convert_img_path(ori_path, suffix):
    new_path = ori_path.replace('clean', suffix)
    assert new_path != ori_path
    return new_path

if __name__ == '__main__':

    pool = Pool(32)
    filelist = []
    for p in corruptions:
        print("\n ### gen corruption:{} ###".format(p))
        for img_path in mmcv.scandir(img_dir, suffix='png', recursive=True):
            srcfile = os.path.join(img_dir, img_path)
            for s in range(5):
                img_suffix = p + "/" + str(s+1)
                out_dir = img_dir.replace('clean', img_suffix)
                assert out_dir != img_dir
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                perturbed_img_path = convert_img_path(srcfile, img_suffix)
                filelist.append((srcfile, p, s+1, perturbed_img_path))
    # import ipdb; ipdb.set_trace()
    pool.map(worker, filelist)