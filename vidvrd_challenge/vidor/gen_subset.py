from vidvrd_challenge.vidor.format.to_ilsvrc_vid_format import *
from vidvrd_challenge.vidor.format.split_video import *
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
    if split == 'val':
        val_root = os.path.join(tgt_ds_root, 'Annotations', 'VID', split)
    else:
        val_root = os.path.join(tgt_ds_root, 'Data', 'VID', split)

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
    print('ImageSets: VID_%s_videos.txt' % split)
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
            video_info = os.path.join('%s/%s/%s %d %d %d\n' % (split, pkg, vid, video_frame_start, 0, n_frame))
            video_frame_start += n_frame
            vid_cnt += 1

            if start:
                val_videos.append(video_info)

        if vid_cnt == vid_end_n:
            break

    val_video_file_path = os.path.join(tgt_imageset_root, 'VID_%s_videos.txt' % split)
    with open(val_video_file_path, 'w') as f:
        f.writelines(val_videos)


def prepare_ImageSets_for_one_video(tgt_ds_root, split, pkg, vid):
    # prepare ImageSets
    tgt_imageset_root = os.path.join(tgt_ds_root, 'ImageSets')
    if not os.path.exists(tgt_imageset_root):
        os.makedirs(tgt_imageset_root)

    if split == 'val':
        val_root = os.path.join(tgt_ds_root, 'Annotations', 'VID', split)
    else:
        val_root = os.path.join(tgt_ds_root, 'Data', 'VID', split)
    vid_path = os.path.join(val_root, pkg, vid)
    n_frame = len(os.listdir(vid_path))

    # 1. VID_val_frames.txt
    print('ImageSets: VID_%s_frames.txt' % split)
    val_frames = []
    for i in range(n_frame):
        frame_info = os.path.join('%s/%s/%s/%06d %d\n' % (split, pkg, vid, i, i+1))
        val_frames.append(frame_info)
    val_frame_file_path = os.path.join(tgt_imageset_root, 'VID_%s_frames.txt' % split)
    with open(val_frame_file_path, 'w') as f:
        f.writelines(val_frames)

    # 2. VID_val_videos.txt
    print('ImageSets: VID_%s_videos.txt' % split)
    val_videos = []
    video_info = os.path.join('%s/%s/%s %d %d %d\n' % (split, pkg, vid, 1, 0, n_frame))
    val_videos.append(video_info)
    val_video_file_path = os.path.join(tgt_imageset_root, 'VID_%s_videos.txt' % split)
    with open(val_video_file_path, 'w') as f:
        f.writelines(val_videos)


def prepare_vidor_gt(tgt_ds_root):

    tgt_imageset_root = os.path.join(tgt_ds_root, 'ImageSets')
    val_video_file_path = os.path.join(tgt_imageset_root, 'VID_val_videos.txt')
    with open(val_video_file_path) as f:
        lines = f.readlines()
        video_ids = [line.split(' ')[0] for line in lines]

    val_anno_root = os.path.join(tgt_ds_root, 'Annotations', 'VID')
    val_data_root = os.path.join(tgt_ds_root, 'Data', 'VID')
    vidor_gt_name = 'vidor_val_object_gt.json'
    gen_vidor_gt(val_anno_root, val_data_root, video_ids, vidor_gt_name)


if __name__ == '__main__':
    tgt_ds_root = '../../data/VidOR-mini'
    tgt_ds_root = os.path.abspath(tgt_ds_root)

    prepare_ImageSets(tgt_ds_root, 'val')
    # prepare_ImageSets_for_one_video(tgt_ds_root, 'val', '0004', '11566980553')
    prepare_vidor_gt(tgt_ds_root)

    # prepare_ImageSets(tgt_ds_root, 'test', 0, 200)

    cache_path = '../../data/cache'
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)




