import os
import json
import xml.etree.ElementTree as ET

ILSVRC_VAL_ROOT = '/home/magus/dataset3/ILSVRC2015/Annotations/VID/val'

vid_val_gt = {}
vid_val_list = sorted(os.listdir(ILSVRC_VAL_ROOT))
for i, vid in enumerate(vid_val_list):
    print('Gen [%d/%d]' % (len(vid_val_list), i+1))
    # for each video
    tid2obj = {}
    vid_frame_dir = os.path.join(ILSVRC_VAL_ROOT, vid)
    vid_frame_list = sorted(os.listdir(vid_frame_dir))
    for f, fid in enumerate(vid_frame_list):
        # for each xml
        frame_path = os.path.join(vid_frame_dir, fid)
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
            obj_traj[str(f)] = [xmin, ymin, xmax, ymax]

    objs = sorted(tid2obj.items(), key=lambda i: i[0])
    gt_objs = []
    for tid, obj in objs:
        gt_obj = {
            'tid': int(tid),
            'category': obj['category'],
            'trajectory': obj['trajectory']
        }
        gt_objs.append(gt_obj)
    vid_val_gt[vid] = gt_objs


with open('imagenet_vid_gt_val_object.json', 'w') as f:
    json.dump(vid_val_gt, f)




