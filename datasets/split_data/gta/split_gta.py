from multiprocessing import Pool
import shutil
import os
shutil._USE_CP_SENDFILE = False
def worker(path_pair):
    srcpath, dstpath = path_pair
    # shutil.copyfile(srcpath, dstpath)
    shutil.move(srcpath, dstpath)

if __name__ == '__main__':
    pool = Pool(32)
    image_path = r'datasets/GTA/images'
    label_path = r'datasets/GTA/labels'

    with open('datasets/split_data/gtav_split_train.txt', 'r') as f:
        train_list = f.readlines()
        train_list = [x.strip() for x in train_list]

    with open('datasets/split_data/gtav_split_val.txt', 'r') as f:
        val_list = f.readlines()
        val_list = [x.strip() for x in val_list]

    with open('datasets/split_data/gtav_split_test.txt', 'r') as f:
        test_list = f.readlines()
        test_list = [x.strip() for x in test_list]

    train_pairs = []

    if not os.path.exists(os.path.join(image_path, 'train')):
        os.makedirs(os.path.join(image_path, 'train'))

    if not os.path.exists(os.path.join(label_path, 'train')):
        os.makedirs(os.path.join(label_path, 'train'))

    for file in train_list:
        srcfile = os.path.join(image_path, file)
        dstfile = os.path.join(image_path, 'train', file)
        train_pairs.append((srcfile, dstfile))

        srclabel = os.path.join(label_path, file)
        dstlabel = os.path.join(label_path, 'train', file)
        train_pairs.append((srclabel, dstlabel))
    pool.map(worker, train_pairs)

    val_pairs = []

    if not os.path.exists(os.path.join(image_path, 'valid')):
        os.makedirs(os.path.join(image_path, 'valid'))

    if not os.path.exists(os.path.join(label_path, 'valid')):
        os.makedirs(os.path.join(label_path, 'valid'))

    for file in val_list:
        srcfile = os.path.join(image_path, file)
        dstfile = os.path.join(image_path, 'valid', file)
        val_pairs.append((srcfile, dstfile))

        srclabel = os.path.join(label_path, file)
        dstlabel = os.path.join(label_path, 'valid', file)
        val_pairs.append((srclabel, dstlabel))
    pool.map(worker, val_pairs)

    test_pairs = []

    if not os.path.exists(os.path.join(image_path, 'test')):
        os.makedirs(os.path.join(image_path, 'test'))

    if not os.path.exists(os.path.join(label_path, 'test')):
        os.makedirs(os.path.join(label_path, 'test'))

    for file in test_list:
        srcfile = os.path.join(image_path, file)
        dstfile = os.path.join(image_path, 'test', file)
        test_pairs.append((srcfile, dstfile))

        srclabel = os.path.join(label_path, file)
        dstlabel = os.path.join(label_path, 'test', file)
        test_pairs.append((srclabel, dstlabel))
    pool.map(worker, test_pairs)