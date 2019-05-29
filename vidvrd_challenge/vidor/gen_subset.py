import json

from vidvrd_challenge.vidor.to_ilsvrc_vid_format import *
from vidvrd_challenge.vidor.split_video import *
from vidvrd_challenge.evaluation.gen_vidor_gt import gen_vidor_gt


def prepare_ImageSets(tgt_ds_root, split, vid_start_n=0, vid_end_n=1000000):
    # prepare ImageSets
    tgt_imageset_root = os.path.join(tgt_ds_root, 'ImageSets')
    if not os.path.exists(tgt_imageset_root):
        os.makedirs(tgt_imageset_root)

    # 1. VID_val_frames.txt
    print('ImageSets: VID_%s_frames.txt' % split)
    val_frames = []
    val_frame_cnt = 1  # start from 1
    vid_cnt = 0
    val_root = os.path.join(tgt_ds_root, 'Annotations', 'VID', split)
    start = False

    for pkg in sorted(os.listdir(val_root)):
        pkg_root = os.path.join(val_root, pkg)

        for vid in sorted(os.listdir(pkg_root)):

            if vid_cnt == vid_start_n:
                start = True
            if vid_cnt == vid_end_n:
                break

            vid_path = os.path.join(pkg_root, vid)
            n_frame = len(os.listdir(vid_path))
            for i in range(n_frame):
                frame_info = os.path.join('%s/%s/%s/%06d %d\n' % (split, pkg, vid, i, val_frame_cnt))
                val_frame_cnt += 1

                if start:
                    val_frames.append(frame_info)

            vid_cnt += 1

        if vid_cnt == vid_end_n:
            break

    val_frame_file_path = os.path.join(tgt_imageset_root, 'VID_%s_frames.txt' % split)
    with open(val_frame_file_path, 'w') as f:
        f.writelines(val_frames)

    # 2. VID_val_videos.txt
    print('ImageSets: VID_val_videos.txt')
    val_videos = []
    video_frame_start = 1  # start from 1
    vid_cnt = 0
    start = False
    for pkg in sorted(os.listdir(val_root)):
        pkg_root = os.path.join(val_root, pkg)

        for vid in sorted(os.listdir(pkg_root)):

            if vid_cnt == vid_start_n:
                start = True
            if vid_cnt == vid_end_n:
                break

            frame_root = os.path.join(pkg_root, vid)

            n_frame = len(os.listdir(frame_root))
            video_info = os.path.join('val/%s/%s %d %d %d\n' % (pkg, vid, video_frame_start, 0, n_frame))
            video_frame_start += n_frame
            vid_cnt += 1

            if start:
                val_videos.append(video_info)

        if vid_cnt == vid_end_n:
            break

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
    vidor_gt_name = 'vidor_val_object_gt.json'
    gen_vidor_gt(val_anno_root, video_ids, vidor_gt_name)


if __name__ == '__main__':
    tgt_ds_root = '../../data/VidOR'
    # prepare_ImageSets(tgt_ds_root, 'val', 10, 30)
    # prepare_vidor_gt(tgt_ds_root)

    prepare_ImageSets(tgt_ds_root, 'test', 0, 200)

    cache_path = '../../data/cache'
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    val_output_path = '../../output/fgfa_rfcn/vidor_vid/resnet_v1_101_flownet_vidor_vid_rfcn_end2end_ohem/VID_val_videos'
    if os.path.exists(val_output_path):
        shutil.rmtree(val_output_path)
    test_output_path = '../../output/fgfa_rfcn/vidor_vid/resnet_v1_101_flownet_vidor_vid_rfcn_end2end_ohem/VID_test_videos'
    if os.path.exists(test_output_path):
        shutil.rmtree(test_output_path)



