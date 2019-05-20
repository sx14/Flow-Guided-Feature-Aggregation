import json

from vidvrd_challenge.vidor.to_ilsvrc_vid_format import *
from vidvrd_challenge.vidor.split_video import *


def prepare_ImageSets(tgt_ds_root, vid_n=10000):
    # prepare ImageSets
    tgt_imageset_root = os.path.join(tgt_ds_root, 'ImageSets')
    if not os.path.exists(tgt_imageset_root):
        os.makedirs(tgt_imageset_root)

    # 1. VID_val_frames.txt
    print('ImageSets: VID_val_frames.txt')
    val_frames = []
    val_frame_cnt = 1  # start from 1
    vid_c = 0
    val_root = os.path.join(tgt_ds_root, 'Annotations', 'VID', 'val')
    for pkg in sorted(os.listdir(val_root)):
        pkg_root = os.path.join(val_root, pkg)

        for vid in sorted(os.listdir(pkg_root)):

            if vid_c >= vid_n:
                break

            vid_path = os.path.join(pkg_root, vid)
            n_frame = len(os.listdir(vid_path))
            for i in range(n_frame):
                frame_info = os.path.join('val/%s/%s/%06d %d\n' % (pkg, vid, i, val_frame_cnt))
                val_frames.append(frame_info)
                val_frame_cnt += 1
            vid_c += 1

        if vid_c >= vid_n:
            break

    val_frame_file_path = os.path.join(tgt_imageset_root, 'VID_val_frames.txt')
    with open(val_frame_file_path, 'w') as f:
        f.writelines(val_frames)

    # 2. VID_val_videos.txt
    print('ImageSets: VID_val_videos.txt')
    val_videos = []
    video_frame_start = 1  # start from 1
    vid_c = 0
    for pkg in sorted(os.listdir(val_root)):
        pkg_root = os.path.join(val_root, pkg)

        for vid in sorted(os.listdir(pkg_root)):

            if vid_c >= vid_n:
                break

            frame_root = os.path.join(pkg_root, vid)

            n_frame = len(os.listdir(frame_root))
            video_info = os.path.join('val/%s/%s %d %d %d\n' % (pkg, vid, video_frame_start, 0, n_frame))
            val_videos.append(video_info)
            video_frame_start += n_frame

        if vid_c >= vid_n:
            break

    val_video_file_path = os.path.join(tgt_imageset_root, 'VID_val_videos.txt')
    with open(val_video_file_path, 'w') as f:
        f.writelines(val_videos)




if __name__ == '__main__':
    tgt_ds_root = '../../data/ILSVRC2015'
    prepare_ImageSets(tgt_ds_root, 10)



