import os
import random
import matplotlib.pyplot as plt
from show_frame import *


def show_video(frame_root, anno_root):
    all_colors = get_colors()
    plt.figure(0)
    print('frame: %d' % len(os.listdir(anno_root)))
    for anno_id in sorted(os.listdir(anno_root)):
        plt.ion()
        plt.axis('off')

        anno_path = os.path.join(anno_root, anno_id)
        frame_path = os.path.join(frame_root, anno_id.split('.')[0]+'.JPEG')
        dets, clss, tids = get_box_cls(anno_path)
        colors = [all_colors[tid] for tid in tids]
        show_boxes(frame_path, dets, clss, colors)

        plt.pause(0.00001)
        plt.cla()

    plt.close()


if __name__ == '__main__':
    data_root = '/media/sunx/Data/linux-workspace/python-workspace/Flow-Guided-Feature-Aggregation/data/VidOR/Data/VID/val/'
    anno_root = '/media/sunx/Data/linux-workspace/python-workspace/Flow-Guided-Feature-Aggregation/data/VidOR/Annotations/VID/val/'
    vid_id = '0004/11566980553'

    video_dir_path = data_root + vid_id
    anno_dir_path = anno_root + vid_id
    show_video(video_dir_path, anno_dir_path)