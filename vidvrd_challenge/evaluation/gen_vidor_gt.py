import os
import json
import xml.etree.ElementTree as ET

import cv2


def gen_vidor_gt(video_anno_root, video_data_root, video_list, save_file_name):

    vid_val_gt = {}
    for i, vid in enumerate(video_list):
        print('Gen [%d/%d]' % (len(video_list), i+1))
        # for each video
        vid_data_frame_dir = os.path.join(video_data_root, vid)
        vid_fst_frame_path = os.path.join(vid_data_frame_dir, '000000.JPEG')
        frame = cv2.imread(vid_fst_frame_path)
        vid_h, vid_w, _ = frame.shape

        tid2obj = {}
        vid_anno_frame_dir = os.path.join(video_anno_root, vid)
        vid_anno_frame_list = sorted(os.listdir(vid_anno_frame_dir))
        for f, fid in enumerate(vid_anno_frame_list):
            # for each xml
            frame_path = os.path.join(vid_anno_frame_dir, fid)
            frame_anno = ET.parse(frame_path)
            frame_objs = frame_anno.findall('object')
            for obj in frame_objs:
                tid = obj.find('trackid').text
                name = obj.find('name').text

                if tid in tid2obj:
                    obj_inst = tid2obj[tid]
                else:
                    obj_inst = {'category': name, 'trajectory': {}}
                    tid2obj[tid] = obj_inst

                obj_box = obj.find('bndbox')
                xmin = int(obj_box.find('xmin').text)
                ymin = int(obj_box.find('ymin').text)
                xmax = int(obj_box.find('xmax').text)
                ymax = int(obj_box.find('ymax').text)

                obj_traj = obj_inst['trajectory']
                obj_traj[fid.split('.')[0]] = [xmin, ymin, xmax, ymax]

        objs = sorted(tid2obj.items(), key=lambda i: i[0])
        gt_objs = []
        for tid, obj in objs:
            gt_obj = {
                'height': vid_h,
                'width': vid_w,
                'tid': int(tid),
                'category': obj['category'],
                'trajectory': obj['trajectory']
            }
            gt_objs.append(gt_obj)

        vid = '/'.join(vid.split('/')[1:])
        vid_val_gt[vid] = gt_objs

    curr_dir = os.path.dirname(__file__)
    save_path = os.path.join(curr_dir, save_file_name)
    with open(save_path, 'w') as f:
        json.dump(vid_val_gt, f)






