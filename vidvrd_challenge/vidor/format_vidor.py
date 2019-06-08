import json

from vidvrd_challenge.vidor.format.to_ilsvrc_vid_format import *
from vidvrd_challenge.vidor.format.split_video import *


def prepare_Data(org_ds_root, tgt_ds_root):
    # extract Data
    org_data_root = os.path.join(org_ds_root, 'vidor')
    tgt_data_root = os.path.join(tgt_ds_root, 'Data', 'VID')
    # org, target
    # splits = [('validation', 'val'), ('training', 'train')]
    splits = [('testing', 'test')]
    for split in splits:
        # target split
        tgt_split_root = os.path.join(tgt_data_root, split[1])
        if not os.path.exists(tgt_split_root):
            os.makedirs(tgt_split_root)

        # original split
        org_split_root = os.path.join(org_data_root, split[0])
        pkgs = sorted(os.listdir(org_split_root))
        for p, pkg in enumerate(pkgs):
            print('Data: [%d/%d]' % (len(pkgs), p+1))
            # original package
            org_pkg_root = os.path.join(org_split_root, pkg)

            # new package
            tgt_pkg_root = os.path.join(tgt_split_root, pkg)
            if not os.path.exists(tgt_pkg_root):
                os.mkdir(tgt_pkg_root)

            for vid in sorted(os.listdir(org_pkg_root)):

                # new frame dir
                video_frame_root = os.path.join(tgt_pkg_root, vid.split('.')[0])
                if not os.path.exists(video_frame_root):
                    os.mkdir(video_frame_root)

                    # load video
                    video_path = os.path.join(org_pkg_root, vid)
                    split_video_ffmpeg(video_path, video_frame_root)


def prepare_ImageSets(tgt_ds_root):
    # prepare ImageSets
    tgt_imageset_root = os.path.join(tgt_ds_root, 'ImageSets')
    if not os.path.exists(tgt_imageset_root):
        os.makedirs(tgt_imageset_root)

    # 1. VID_val_frames.txt
    # print('ImageSets: VID_val_frames.txt')
    # val_frames = []
    # val_frame_cnt = 1   # start from 1
    # val_root = os.path.join(tgt_ds_root, 'Annotations', 'VID', 'val')
    # for pkg in os.listdir(val_root):
    #     pkg_root = os.path.join(val_root, pkg)
    #
    #     for vid in os.listdir(pkg_root):
    #         vid_path = os.path.join(pkg_root, vid)
    #
    #         n_frame = len(os.listdir(vid_path))
    #         for i in range(n_frame):
    #             frame_info = os.path.join('val/%s/%s/%06d %d\n' % (pkg, vid, i, val_frame_cnt))
    #             val_frames.append(frame_info)
    #             val_frame_cnt += 1
    #
    # val_frame_file_path = os.path.join(tgt_imageset_root, 'VID_val_frames.txt')
    # with open(val_frame_file_path, 'w') as f:
    #     f.writelines(val_frames)

    # 2. VID_val_videos.txt
    # print('ImageSets: VID_val_videos.txt')
    # val_videos = []
    # video_frame_start = 1   # start from 1
    # for pkg in os.listdir(val_root):
    #     pkg_root = os.path.join(val_root, pkg)
    #
    #     for vid in os.listdir(pkg_root):
    #         frame_root = os.path.join(pkg_root, vid)
    #
    #         n_frame = len(os.listdir(frame_root))
    #         video_info = os.path.join('val/%s/%s %d %d %d\n' % (pkg, vid, video_frame_start, 0, n_frame))
    #         val_videos.append(video_info)
    #         video_frame_start += n_frame
    #
    # val_video_file_path = os.path.join(tgt_imageset_root, 'VID_val_videos.txt')
    # with open(val_video_file_path, 'w') as f:
    #     f.writelines(val_videos)

    # 3. VID_train_15frames.txt
    # print('ImageSets: VID_train_15frames.txt')
    # train_key_frames = []
    # train_root = os.path.join(tgt_ds_root, 'Annotations', 'VID', 'train')
    # n_seg = 35  # TODO: need tune
    # n_seg_frame = 20
    # # n_frm_max = 900
    # for pkg in os.listdir(train_root):
    #     pkg_root = os.path.join(train_root, pkg)
    #
    #     for vid in os.listdir(pkg_root):
    #         frame_root = os.path.join(pkg_root, vid)
    #         n_frame = len(os.listdir(frame_root))
    #         # n_frame = min(n_frame, n_frm_max)
    #
    #         # n_seg_frame = max(n_frame * 1.0 / n_seg, 1.0)
    #         key_frame_id = int(n_seg_frame / 2.0)
    #         while key_frame_id <= (n_frame-1):
    #             key_frame_info = os.path.join('train/%s/%s %d %d %d\n' % (pkg, vid, 1, int(key_frame_id), n_frame))
    #             train_key_frames.append(key_frame_info)
    #             key_frame_id += n_seg_frame
    #
    # train_key_frame_file_path = os.path.join(tgt_imageset_root, 'VID_train_15frames.txt')
    # with open(train_key_frame_file_path, 'w') as f:
    #     f.writelines(train_key_frames)


    # 4. VID_test_frames.txt
    print('ImageSets: VID_test_frames.txt')
    val_frames = []
    val_frame_cnt = 1   # start from 1
    val_root = os.path.join(tgt_ds_root, 'Data', 'VID', 'test')
    for pkg in os.listdir(val_root):
        pkg_root = os.path.join(val_root, pkg)

        for vid in os.listdir(pkg_root):
            vid_path = os.path.join(pkg_root, vid)

            n_frame = len(os.listdir(vid_path))
            for i in range(n_frame):
                frame_info = os.path.join('test/%s/%s/%06d %d\n' % (pkg, vid, i, val_frame_cnt))
                val_frames.append(frame_info)
                val_frame_cnt += 1

    val_frame_file_path = os.path.join(tgt_imageset_root, 'VID_test_frames.txt')
    with open(val_frame_file_path, 'w') as f:
        f.writelines(val_frames)

    # 5. VID_test_videos.txt
    print('ImageSets: VID_test_videos.txt')
    val_videos = []
    video_frame_start = 1   # start from 1
    for pkg in os.listdir(val_root):
        pkg_root = os.path.join(val_root, pkg)

        for vid in os.listdir(pkg_root):
            frame_root = os.path.join(pkg_root, vid)

            n_frame = len(os.listdir(frame_root))
            video_info = os.path.join('test/%s/%s %d %d %d\n' % (pkg, vid, video_frame_start, 0, n_frame))
            val_videos.append(video_info)
            video_frame_start += n_frame

    val_video_file_path = os.path.join(tgt_imageset_root, 'VID_test_videos.txt')
    with open(val_video_file_path, 'w') as f:
        f.writelines(val_videos)


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

                # for each frame
                vid_frame_objs = vid_anno['trajectories']
                anno_frame_n = len(vid_frame_objs)

                if data_frame_n != anno_frame_n:
                    print('[WARNING]%s: A(%d) | F(%d)' % (anno_frame_root, anno_frame_n, data_frame_n))

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


def prepare_Annotations_test(tgt_ds_root):
    data_root = os.path.join(org_ds_root, 'Data', 'VID', 'test')
    anno_root = os.path.join(tgt_ds_root, 'Annotations', 'VID', 'test')

    # original path
    pkgs = sorted(os.listdir(data_root))
    for p, pkg in enumerate(pkgs):
        print('Annotations test: [%d/%d]' % (len(pkgs), p + 1))
        pkg_root = os.path.join(data_root, pkg)

        # new package
        tgt_pkg_root = os.path.join(anno_root, pkg)
        if not os.path.exists(tgt_pkg_root):
            os.mkdir(tgt_pkg_root)

        for vid in sorted(os.listdir(pkg_root)):
            # org video annotation
            vid_frame0_path = os.path.join(pkg_root, vid, '000000.JPEG')
            vid_frame0 = cv2.imread(vid_frame0_path)
            vid_height, vid_width, _ = vid_frame0.shape

            # frame annotation dir
            anno_frame_root = os.path.join(tgt_pkg_root, vid)
            if not os.path.exists(anno_frame_root):
                os.mkdir(anno_frame_root)

            vid_frame_dir = os.path.join(pkg_root, vid)
            for f in range(len(os.listdir(vid_frame_dir))):
                mid_anno = dict()
                mid_anno['folder'] = '%s/%s' % (pkg, vid)
                mid_anno['width'] = vid_width
                mid_anno['height'] = vid_height
                mid_anno['database'] = 'VidOR'
                mid_anno['filename'] = '%06d' % f
                mid_objs = []

                mid_obj = dict()
                tid = 0
                mid_obj['trackid'] = tid
                mid_obj['xmax'] = 100
                mid_obj['ymax'] = 100
                mid_obj['xmin'] = 50
                mid_obj['ymin'] = 50
                mid_obj['generated'] = 0
                mid_obj['tracker'] = 'none'
                mid_obj['name'] = 'adult'
                mid_objs.append(mid_obj)

                mid_anno['objects'] = mid_objs

                output_path = os.path.join(anno_frame_root, mid_anno['filename']+'.xml')
                output_ilsvrc_vid_format(mid_anno, output_path)


def collect_frame_error(org_ds_root, tgt_ds_root):
    inconsistent_videos = ['video_id AnnoFrameN VidFrameN\n']

    org_anno_root = os.path.join(org_ds_root, 'annotation')
    tgt_anno_root = os.path.join(tgt_ds_root, 'Data', 'VID')

    # (org split, target split)
    splits = [('validation', 'val'), ('training', 'train')]
    for split in splits:
        tgt_split_root = os.path.join(tgt_anno_root, split[1])
        org_split_root = os.path.join(org_anno_root, split[0])

        pkgs = sorted(os.listdir(org_split_root))
        for p, pkg in enumerate(pkgs):
            print('Collect: [%d/%d]' % (len(pkgs), p + 1))
            org_pkg_root = os.path.join(org_split_root, pkg)
            tgt_pkg_root = os.path.join(tgt_split_root, pkg)

            for vid in sorted(os.listdir(org_pkg_root)):
                # org video annotation
                vid_anno_path = os.path.join(org_pkg_root, vid)
                vid_anno = json.load(open(vid_anno_path))
                vid_frame_objs = vid_anno['trajectories']
                anno_frame_n = len(vid_frame_objs)

                # frame annotation dir
                data_frame_root = os.path.join(tgt_pkg_root, vid.split('.')[0])
                data_frame_n = len(os.listdir(data_frame_root))

                if data_frame_n != anno_frame_n:
                    inconsistent_videos.append('%s/%s/%s %d %d\n' % (split[0], pkg, vid, anno_frame_n, data_frame_n))

    with open('video_frame_inconsistency.txt', 'w') as f:
        f.writelines(inconsistent_videos)


def collect_category_error(org_ds_root):
    classes = ['__background__',  # always index 0
               'bread', 'cake', 'dish', 'fruits',
               'vegetables', 'backpack', 'camera', 'cellphone',
               'handbag', 'laptop', 'suitcase', 'ball/sports_ball',
               'bat', 'frisbee', 'racket', 'skateboard',
               'ski', 'snowboard', 'surfboard', 'toy',
               'baby_seat', 'bottle', 'chair', 'cup',
               'electric_fan', 'faucet', 'microwave', 'oven',
               'refrigerator', 'screen/monitor', 'sink', 'sofa',
               'stool', 'table', 'toilet', 'guitar',
               'piano', 'baby_walker', 'bench', 'stop_sign',
               'traffic_light', 'aircraft', 'bicycle', 'bus/truck',
               'car', 'motorcycle', 'scooter', 'train',
               'watercraft', 'crab', 'bird', 'chicken',
               'duck', 'penguin', 'fish', 'stingray',
               'crocodile', 'snake', 'turtle', 'antelope',
               'bear', 'camel', 'cat', 'cattle/cow',
               'dog', 'elephant', 'hamster/rat', 'horse',
               'kangaroo', 'leopard', 'lion', 'panda',
               'pig', 'rabbit', 'sheep/goat', 'squirrel',
               'tiger', 'adult', 'baby', 'child']

    org_anno_root = os.path.join(org_ds_root, 'annotation')
    categories = set()
    unseen_categories = set()

    # (org split, target split)
    splits = [('validation', 'val'), ('training', 'train')]
    for split in splits:
        org_split_root = os.path.join(org_anno_root, split[0])

        pkgs = sorted(os.listdir(org_split_root))
        for p, pkg in enumerate(pkgs):
            print('Collect: [%d/%d]' % (len(pkgs), p + 1))
            org_pkg_root = os.path.join(org_split_root, pkg)

            for vid in sorted(os.listdir(org_pkg_root)):
                # org video annotation
                vid_anno_path = os.path.join(org_pkg_root, vid)
                vid_anno = json.load(open(vid_anno_path))
                vid_obj_clss = vid_anno['subject/objects']

                for obj_cls in vid_obj_clss:
                    categories.add(obj_cls['category'])
                    if obj_cls['category'] not in classes:
                        unseen_categories.add(obj_cls['category'])
                        print(obj_cls['category'])
    print('>>>>>>>>>> category <<<<<<<<<<')
    for c in categories:
        print(c)

    print('>>>>>>>>>> unseen category <<<<<<<<<<')
    for c in unseen_categories:
        print(c)

if __name__ == '__main__':
    org_ds_root = '/home/magus/dataset3/VidOR/vidor-dataset'
    tgt_ds_root = '/home/magus/dataset3/VidOR/vidor-ilsvrc'
    #prepare_Data(org_ds_root, tgt_ds_root)
    #prepare_Annotations(org_ds_root, tgt_ds_root)
    #prepare_Annotations_test(tgt_ds_root)
    #prepare_ImageSets(tgt_ds_root)

    # collect_frame_error(org_ds_root, tgt_ds_root)
    # collect_category_error(org_ds_root)


