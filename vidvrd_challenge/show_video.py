import os
import random
import matplotlib.pyplot as plt
from show_frame import *

def show_video(frame_root, anno_root):
    for anno_id in sorted(os.listdir(anno_root)):
        anno_path = os.path.join(anno_root, anno_id)
        frame_path = os.path.join(frame_root, anno_id.split('.')[0]+'.JPEG')
        dets, clss = get_box_cls(anno_path)
        show_boxes(frame_path, dets, clss, 'mul')


if __name__ == '__main__':
    data_root = '/home/magus/sunx-workspace/project/Flow-Guided-Feature-Aggregation/data/VidOR/Data/VID/train/'
    anno_root = '/home/magus/sunx-workspace/project/Flow-Guided-Feature-Aggregation/data/VidOR/Annotations/VID/train/'
    vid_id = '0000/2401075277'

    video_dir_path = data_root + vid_id
    anno_dir_path = anno_root + vid_id
    show_video(video_dir_path, anno_dir_path)