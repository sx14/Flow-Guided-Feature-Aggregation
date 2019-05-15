import os
import json

import cv2

from vidvrd_challenge.to_ilsvrc_vid_format import *
from vidvrd_challenge.split_video import *


def prepare_Data(org_ds_root, tgt_ds_root):
    # extract Data
    org_data_root = os.path.join(org_ds_root, 'vidor')
    tgt_data_root = os.path.join(tgt_ds_root, 'Data1', 'VID')
    # org, target
    splits = [('validation', 'val'), ('training', 'train')]
    for split in splits:
        # target path
        tgt_split_root = os.path.join(tgt_data_root, split[1])
        if not os.path.exists(tgt_split_root):
            os.makedirs(tgt_split_root)

        # original path
        org_split_root = os.path.join(org_data_root, split[0])
        pkgs = sorted(os.listdir(org_split_root))
        for p, pkg in enumerate(pkgs):
            print('Data: [%d/%d]' % (len(pkgs), p+1))
            org_pkg_root = os.path.join(org_split_root, pkg)

            # new package
            tgt_pkg_root = os.path.join(tgt_split_root, pkg)
            if not os.path.exists(tgt_pkg_root):
                os.mkdir(tgt_pkg_root)

            for vid in sorted(os.listdir(org_pkg_root)):

                # frame dir
                video_frame_root = os.path.join(tgt_pkg_root, vid.split('.')[0])
                if not os.path.exists(video_frame_root):
                    os.mkdir(video_frame_root)

                    # load video
                    video_path = os.path.join(org_pkg_root, vid)
                    # split_video_cv2(video_path, video_frame_root)
                    split_video_ffmpeg(video_path, video_frame_root)


def prepare_ImageSets(tgt_ds_root):
    # prepare ImageSets
    tgt_imageset_root = os.path.join(tgt_ds_root, 'ImageSets')
    if not os.path.exists(tgt_imageset_root):
        os.makedirs(tgt_imageset_root)

    # 1. VID_val_frames.txt
    print('ImageSets: VID_val_frames.txt')
    val_frames = []
    val_frame_cnt = 1   # start from 1
    val_root = os.path.join(tgt_ds_root, 'Annotations', 'VID', 'val')
    for pkg in os.listdir(val_root):
        pkg_root = os.path.join(val_root, pkg)

        for vid in os.listdir(pkg_root):
            vid_path = os.path.join(pkg_root, vid)

            n_frame = len(os.listdir(vid_path))
            for i in range(n_frame):
                frame_info = os.path.join('val/%s/%s/%06d %d\n' % (pkg, vid, i, val_frame_cnt))
                val_frames.append(frame_info)
                val_frame_cnt += 1

    val_frame_file_path = os.path.join(tgt_imageset_root, 'VID_val_frames.txt')
    with open(val_frame_file_path, 'w') as f:
        f.writelines(val_frames)

    # 2. VID_val_videos.txt
    print('ImageSets: VID_val_videos.txt')
    val_videos = []
    video_frame_start = 1   # start from 1
    for pkg in os.listdir(val_root):
        pkg_root = os.path.join(val_root, pkg)

        for vid in os.listdir(pkg_root):
            frame_root = os.path.join(pkg_root, vid)

            n_frame = len(os.listdir(frame_root))
            video_info = os.path.join('val/%s/%s %d %d %d\n' % (pkg, vid, video_frame_start, 0, n_frame))
            val_videos.append(video_info)
            video_frame_start += n_frame

    val_video_file_path = os.path.join(tgt_imageset_root, 'VID_val_videos.txt')
    with open(val_video_file_path, 'w') as f:
        f.writelines(val_videos)

    # 3. VID_train_15frames.txt
    print('ImageSets: VID_train_15frames.txt')
    train_key_frames = []
    train_root = os.path.join(tgt_ds_root, 'Annotations', 'VID', 'train')
    n_seg = 35  # TODO: need tune
    n_frm_max = 900
    for pkg in os.listdir(train_root):
        pkg_root = os.path.join(train_root, pkg)

        for vid in os.listdir(pkg_root):
            frame_root = os.path.join(pkg_root, vid)
            n_frame = len(os.listdir(frame_root))
            n_frame = min(n_frame, n_frm_max)

            n_seg_frame = max(n_frame * 1.0 / n_seg, 1.0)
            key_frame_id = int(n_seg_frame / 2.0)
            while key_frame_id <= (n_frame-1):
                key_frame_info = os.path.join('train/%s/%s %d %d %d\n' % (pkg, vid, 1, int(key_frame_id), n_frame))
                train_key_frames.append(key_frame_info)
                key_frame_id += n_seg_frame

    train_key_frame_file_path = os.path.join(tgt_imageset_root, 'VID_train_15frames.txt')
    with open(train_key_frame_file_path, 'w') as f:
        f.writelines(train_key_frames)


def prepare_Annotations(org_ds_root, tgt_ds_root):
    org_anno_root = os.path.join(org_ds_root, 'annotation')
    tgt_anno_root = os.path.join(tgt_ds_root, 'Annotations', 'VID')
    # org, target
    splits = [('validation', 'val'), ('training', 'train')]
    for split in splits:
        # target path
        tgt_split_root = os.path.join(tgt_anno_root, split[1])
        if not os.path.exists(tgt_split_root):
            os.makedirs(tgt_split_root)

        # original path
        org_split_root = os.path.join(org_anno_root, split[0])
        pkgs = sorted(os.listdir(org_split_root))
        for p, pkg in enumerate(pkgs):
            print('Annotations: [%d/%d]' % (len(pkgs), p + 1))
            org_pkg_root = os.path.join(org_split_root, pkg)

            # new package
            tgt_pkg_root = os.path.join(tgt_split_root, pkg)
            if not os.path.exists(tgt_pkg_root):
                os.mkdir(tgt_pkg_root)

            for vid in sorted(os.listdir(org_pkg_root)):
                # org video annotation
                vid_anno_path = os.path.join(org_pkg_root, vid)
                vid_anno = json.load(open(vid_anno_path))
                vid_width = vid_anno['width']
                vid_height = vid_anno['height']
                vid_obj_clss = vid_anno['subject/objects']
                tid2cls = dict()
                for obj_cls in vid_obj_clss:
                    tid2cls[obj_cls['tid']] = obj_cls['category']

                # frame annotation dir
                anno_frame_root = os.path.join(tgt_pkg_root, vid.split('.')[0])
                if not os.path.exists(anno_frame_root):
                    os.mkdir(anno_frame_root)

                data_frame_root = anno_frame_root.replace('Annotations', 'Data')
                data_frame_n = len(os.listdir(data_frame_root))

                data_frame_root1 = anno_frame_root.replace('Annotations', 'Data1')
                data_frame_n1 = len(os.listdir(data_frame_root1))

                # for each frame
                vid_frame_objs = vid_anno['trajectories']
                anno_frame_n = len(vid_frame_objs)


                #if data_frame_n != anno_frame_n:
                print('[WARNING]%s: A(%d) | cv2(%d) | ffmpeg(%d)' % (anno_frame_root, anno_frame_n, data_frame_n, data_frame_n1))

                for f in range(len(vid_frame_objs)):
                    mid_anno = dict()
                    mid_anno['folder'] = '%s/%s' % (pkg, vid.split('.')[0])
                    mid_anno['width'] = vid_width
                    mid_anno['height'] = vid_height
                    mid_anno['database'] = 'VidOR'
                    mid_anno['filename'] = '%06d' % f
                    mid_objs = []
                    for obj in vid_frame_objs[f]:
                        mid_obj = dict()
                        tid = obj['tid']
                        mid_obj['trackid'] = tid
                        mid_obj['xmax'] = obj['bbox']['xmax']
                        mid_obj['ymax'] = obj['bbox']['ymax']
                        mid_obj['xmin'] = obj['bbox']['xmin']
                        mid_obj['ymin'] = obj['bbox']['ymin']
                        mid_obj['generated'] = obj['generated']
                        mid_obj['tracker'] = obj['tracker']
                        mid_obj['name'] = tid2cls[tid]
                        mid_objs.append(mid_obj)
                    mid_anno['objects'] = mid_objs

                    output_path = os.path.join(anno_frame_root, mid_anno['filename']+'.xml')
                    output_ilsvrc_vid_format(mid_anno, output_path)


if __name__ == '__main__':
    org_ds_root = '/home/magus/dataset3/VidOR/vidor-dataset'
    tgt_ds_root = '/home/magus/dataset3/VidOR/vidor-ilsvrc'
    # prepare_Data(org_ds_root, tgt_ds_root)
    prepare_Annotations(org_ds_root, tgt_ds_root)
    #prepare_ImageSets(tgt_ds_root)



