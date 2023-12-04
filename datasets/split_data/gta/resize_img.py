import os
from PIL import Image
import numpy as np
import cv2

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def resize_split(split):
  filenames = GetFileFromThisRootDir(f'datasets/GTA/images/{split}')
  for filename in filenames:
    basename = os.path.basename(filename)
    img = Image.open(filename)
    gtname = os.path.join(f'datasets/GTA/labels/{split}', basename)
    gt = Image.open(gtname)
    print(f'filename: {filename}')
    if not os.path.exists(f'datasets/GTA/labels/{split}_resize'):
      os.makedirs(f'datasets/GTA/labels/{split}_resize')
    if (img.width != gt.width) or (img.height != gt.height):
      # read img
      gt_np = np.asarray(gt)
      # resize img
      width, height = img.width, img.height
      resized_gt_np = cv2.resize(gt_np, (width, height), interpolation=cv2.INTER_NEAREST)
      # import ipdb;
      # ipdb.set_trace()
      # save img
      outname = os.path.join(f'datasets/GTA/labels/{split}_resize', basename)
      cv2.imwrite(outname, resized_gt_np)

if __name__ == '__main__':
  resize_split('valid')
  # resize_split('train')
  # resize_split('test')