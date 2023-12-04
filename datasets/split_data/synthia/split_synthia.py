from multiprocessing import Pool
import shutil
import os
shutil._USE_CP_SENDFILE = False

def worker(path_pair):
    srcpath, dstpath = path_pair
    # print(f'srcpath{srcpath}')
    # print(f'dstpath{dstpath}')
    # shutil.copyfile(srcpath, dstpath)
    shutil.move(srcpath, dstpath)

if __name__ == '__main__':
    pool = Pool(32)
    image_path = r'datasets/synthia/RGB'
    label_path = r'datasets/synthia/GT/LABELS'

    # dst_image_path = r'datasets/synthia_split/RGB'
    # dst_label_path = r'datasets/synthia_split/GT'

    dst_image_path = image_path
    dst_label_path = label_path

    with open('datasets/split_data/synthia_split_train.txt', 'r') as f:
        train_list = f.readlines()
        train_list = [x.strip() for x in train_list]

    with open('datasets/split_data/synthia_split_val.txt', 'r') as f:
        val_list = f.readlines()
        val_list = [x.strip() for x in val_list]

    train_pairs = []

    if not os.path.exists(os.path.join(dst_image_path, 'train')):
        os.makedirs(os.path.join(dst_image_path, 'train'))

    if not os.path.exists(os.path.join(dst_label_path, 'train')):
        os.makedirs(os.path.join(dst_label_path, 'train'))

    for file in train_list:
        srcfile = os.path.join(image_path, file)
        dstfile = os.path.join(dst_image_path, 'train', file)
        train_pairs.append((srcfile, dstfile))

        srclabel = os.path.join(label_path, file)
        dstlabel = os.path.join(dst_label_path, 'train', file)
        train_pairs.append((srclabel, dstlabel))
    pool.map(worker, train_pairs)

    val_pairs = []

    if not os.path.exists(os.path.join(dst_image_path, 'val')):
        os.makedirs(os.path.join(dst_image_path, 'val'))

    if not os.path.exists(os.path.join(dst_label_path, 'val')):
        os.makedirs(os.path.join(dst_label_path, 'val'))

    for file in val_list:
        srcfile = os.path.join(image_path, file)
        dstfile = os.path.join(dst_image_path, 'val', file)
        val_pairs.append((srcfile, dstfile))

        srclabel = os.path.join(label_path, file)
        dstlabel = os.path.join(dst_label_path, 'val', file)
        val_pairs.append((srclabel, dstlabel))
    pool.map(worker, val_pairs)
