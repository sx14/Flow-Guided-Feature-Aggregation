import os
import cv2

org_ds_root = ''
tgt_ds_root = ''


# extract frames
tgt_data_root = os.path.join(tgt_ds_root, 'Data', 'VID')
org_data_root = os.path.join(org_ds_root, 'vidor')
splits = [('training', 'train'), ('validation', 'val')]
for split in splits:
    # target path
    tgt_split_root = os.path.join(tgt_data_root, split[1])
    if not os.path.exists(tgt_split_root):
        os.makedirs(tgt_split_root)

    # original path
    org_split_root = os.path.join(org_data_root, split[0])
    for pkg in os.listdir(org_split_root):
        org_pkg_root = os.path.join(org_split_root, pkg)

        tgt_pkg_root = os.path.join(tgt_split_root, pkg)
        os.mkdir(tgt_pkg_root)

        for vid in os.listdir(org_pkg_root):
            vid_path = os.path.join(org_pkg_root, vid)
            print('read video:' + vid_path)
            # load video
            video = cv2.VideoCapture(vid_path)
            has_next = video.isOpened()
            assert has_next

            video_frame_root = os.path.join(tgt_pkg_root, vid.split('.')[0])
            os.mkdir(video_frame_root)

            fid = 0
            while has_next:
                has_next, frame = video.read()
                frame_path = os.path.join(video_frame_root, '%06d.JPEG' % fid)
                cv2.imwrite(frame_path, frame)
                print('save frame: %d' % fid)


# prepare annotations
os.makedirs(os.path.join(tgt_ds_root, 'Annotations', 'VID'))

