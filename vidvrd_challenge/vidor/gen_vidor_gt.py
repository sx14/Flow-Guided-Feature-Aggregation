import os
import json
import xml.etree.ElementTree as ET

ILSVRC_VAL_ROOT = '../../data/VidOR/Annotations/VID/val'

vid_val_gt = {}
pkg_list = sorted(os.listdir(ILSVRC_VAL_ROOT))
for p, pkg_id in enumerate(pkg_list):
    print('Gen [%d/%d]' % (len(pkg_list), p + 1))

    pkg_dir = os.path.join(ILSVRC_VAL_ROOT, pkg_id)
    vid_list = sorted(os.listdir(pkg_dir))
    for i, vid in enumerate(vid_list):
        # for each video
        tid2obj = {}
        vid_frame_dir = os.path.join(pkg_dir, vid)
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
                'trajectory': obj['trajectory']}
            gt_objs.append(gt_obj)
        vid_val_gt[vid] = gt_objs


with open('../evaluation/vidor_val_object_gt.json', 'w') as f:
    json.dump(vid_val_gt, f)




