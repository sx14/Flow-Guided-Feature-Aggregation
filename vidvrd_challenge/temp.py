import json

from vidvrd_challenge.vidor.to_ilsvrc_vid_format import *
from vidvrd_challenge.vidor.split_video import *





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
    for vid in sorted(os.listdir(val_root)):
        vid_path = os.path.join(val_root, vid)

        n_frame = len(os.listdir(vid_path))
        for i in range(n_frame):
            frame_info = os.path.join('val/%s/%06d %d\n' % (vid, i, val_frame_cnt))
            val_frames.append(frame_info)
            val_frame_cnt += 1

    val_frame_file_path = os.path.join(tgt_imageset_root, 'VID_val_frames.txt')
    with open(val_frame_file_path, 'w') as f:
        f.writelines(val_frames)

    # 2. VID_val_videos.txt
    print('ImageSets: VID_val_videos.txt')
    val_videos = []
    video_frame_start = 1   # start from 1
    for vid in os.listdir(val_root):
        frame_root = os.path.join(val_root, vid)

        n_frame = len(os.listdir(frame_root))
        video_info = os.path.join('val/%s %d %d %d\n' % (vid, video_frame_start, 0, n_frame))
        val_videos.append(video_info)
        video_frame_start += n_frame

    val_video_file_path = os.path.join(tgt_imageset_root, 'VID_val_videos.txt')
    with open(val_video_file_path, 'w') as f:
        f.writelines(val_videos)




if __name__ == '__main__':
    tgt_ds_root = '../data/ILSVRC2015'
    prepare_ImageSets(tgt_ds_root)



