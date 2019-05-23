import json
import shutil

from vidvrd_challenge.vidor.to_ilsvrc_vid_format import *
from vidvrd_challenge.vidor.split_video import *
from vidvrd_challenge.evaluation.gen_vidor_gt import gen_vidor_gt


def prepare_ImageSets(tgt_ds_root, vid_start_n=0, vid_end_n=1000000):
    # prepare ImageSets
    tgt_imageset_root = os.path.join(tgt_ds_root, 'ImageSets')
    if not os.path.exists(tgt_imageset_root):
        os.makedirs(tgt_imageset_root)

    # 1. VID_val_frames.txt
    print('ImageSets: VID_val_frames.txt')
    val_frames = []
    val_frame_cnt = 1   # start from 1
    val_root = os.path.join(tgt_ds_root, 'Annotations', 'VID', 'val')
    start = False
    for i, vid in enumerate(sorted(os.listdir(val_root))):

        if i == vid_start_n:
            start = True
        if i == vid_end_n:
            break

        vid_path = os.path.join(val_root, vid)
        n_frame = len(os.listdir(vid_path))
        for i in range(n_frame):
            frame_info = os.path.join('val/%s/%06d %d\n' % (vid, i, val_frame_cnt))
            val_frame_cnt += 1
            if start:
                val_frames.append(frame_info)

    val_frame_file_path = os.path.join(tgt_imageset_root, 'VID_val_frames.txt')
    with open(val_frame_file_path, 'w') as f:
        f.writelines(val_frames)

    # 2. VID_val_videos.txt
    print('ImageSets: VID_val_videos.txt')
    val_videos = []
    video_frame_start = 1   # start from 1
    start = False
    for i, vid in enumerate(sorted(os.listdir(val_root))):

        if i == vid_start_n:
            start = True
        if i == vid_end_n:
            break

        frame_root = os.path.join(val_root, vid)
        n_frame = len(os.listdir(frame_root))
        video_info = os.path.join('val/%s %d %d %d\n' % (vid, video_frame_start, 0, n_frame))
        video_frame_start += n_frame

        if start:
            val_videos.append(video_info)

    val_video_file_path = os.path.join(tgt_imageset_root, 'VID_val_videos.txt')
    with open(val_video_file_path, 'w') as f:
        f.writelines(val_videos)


def prepare_vidor_gt(tgt_ds_root):

    tgt_imageset_root = os.path.join(tgt_ds_root, 'ImageSets')
    val_video_file_path = os.path.join(tgt_imageset_root, 'VID_val_videos.txt')
    with open(val_video_file_path) as f:
        lines = f.readlines()
        video_ids = [line.split(' ')[0] for line in lines]

    val_anno_root = tgt_ds_root + '/Annotations/VID'
    vidor_gt_name = 'imagenet_val_object_gt.json'
    gen_vidor_gt(val_anno_root, video_ids, vidor_gt_name)


if __name__ == '__main__':
    tgt_ds_root = '../../data/ILSVRC2015'
    prepare_ImageSets(tgt_ds_root, 0, 2)
    prepare_vidor_gt(tgt_ds_root)

    # shutil.rmtree('../../data/cache')
    # shutil.rmtree('../../output/fgfa_rfcn/imagenet_vid/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem/VID_val_videos')



