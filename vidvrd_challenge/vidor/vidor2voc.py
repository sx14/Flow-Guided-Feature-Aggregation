import json

from vidvrd_challenge.vidor.format.to_ilsvrc_vid_format import *
from vidvrd_challenge.vidor.format.split_video import *


def prepare_ImageSets(tgt_ds_root):

    # prepare train.txt
    train_frame_path = os.path.join(tgt_ds_root, 'ImageSets', 'VID_train_15frames.txt')
    with open(train_frame_path) as f:
        train_items = f.readlines()
        train_items = [item.strip().split(' ') for item in train_items]
        train_items = ['%s/%06d\n' % (train_items[i][0], int(train_items[i][2])) for i in range(len(train_items))]

    train_list_dir = os.path.join(tgt_ds_root, 'ImageSets', 'Main')
    if not os.path.exists(train_list_dir):
        os.makedirs(train_list_dir)
    train_list_path = os.path.join(train_list_dir, 'train.txt')
    with open(train_list_path, 'w') as f:
        f.writelines(train_items)

    # prepare val.txt
    val_frame_path = os.path.join(tgt_ds_root, 'ImageSets', 'VID_val_frames.txt')
    with open(val_frame_path) as f:
        val_items = f.readlines()
        val_items = [item.strip().split(' ') for item in val_items]
        val_items = ['%s\n' % (val_items[i][0]) for i in range(0, len(val_items), 20)]

    val_list_dir = os.path.join(tgt_ds_root, 'ImageSets', 'Main')
    if not os.path.exists(val_list_dir):
        os.makedirs(val_list_dir)
    val_list_path = os.path.join(val_list_dir, 'val.txt')
    with open(val_list_path, 'w') as f:
        f.writelines(val_items)


    # prepare test.txt
    test_frame_path = os.path.join(tgt_ds_root, 'ImageSets', 'VID_test_frames.txt')
    with open(test_frame_path) as f:
        test_items = f.readlines()
        test_items = [item.strip().split(' ') for item in test_items]
        test_items = ['%s\n' % (val_items[i][0]) for i in range(0, len(test_items), 20)]

    test_list_dir = os.path.join(tgt_ds_root, 'ImageSets', 'Main')
    if not os.path.exists(test_list_dir):
        os.makedirs(test_list_dir)
    test_list_path = os.path.join(test_list_dir, 'test.txt')
    with open(test_list_path, 'w') as f:
        f.writelines(test_items)


if __name__ == '__main__':
    tgt_ds_root = '/home/magus/sunx-workspace/datasets/vidor/vidor-ilsvrc'
    prepare_ImageSets(tgt_ds_root)



