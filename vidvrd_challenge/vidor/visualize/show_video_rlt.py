import os
import json
import random
import matplotlib.pyplot as plt
from show_frame import *


def read_anno(anno_path):
    with open(anno_path) as f:
        vid_anno = json.load(f)

    objs = {}
    n_frame = vid_anno['frame_count']
    vid_obj_clss = vid_anno['subject/objects']
    tid2cls = dict()
    for obj_cls in vid_obj_clss:
        tid2cls[obj_cls['tid']] = obj_cls['category']
        objs[obj_cls['tid']] = [None for _ in range(n_frame)]

    # object trajs
    frame_boxes = vid_anno['trajectories']
    for f in range(len(frame_boxes)):
        boxes = frame_boxes[f]
        for box in boxes:
            tid = box['tid']
            xmin = box['bbox']['xmin']
            ymin = box['bbox']['ymin']
            xmax = box['bbox']['xmax']
            ymax = box['bbox']['ymax']
            w = xmax-xmin+1
            h = ymax-ymin+1
            objs[tid][f] = [xmin, ymin, w, h]

    # relations
    relations = vid_anno['relation_instances']
    return objs, relations, tid2cls


def show_video(frame_root, anno_path):
    all_colors = get_colors()
    objs, relations, tid2cls = read_anno(anno_path)
    frame_num = len(objs[0])

    for i, rlt in enumerate(relations):

        stt_fid = rlt['begin_fid']
        end_fid = rlt['end_fid']
        sbj_tid = rlt['subject_tid']
        obj_tid = rlt['object_tid']
        colors0 = [all_colors[sbj_tid], all_colors[obj_tid]]
        predicate = rlt['predicate']


        print('R[%d/%d] %s [0| %d -> %d |%d]' % (len(relations), i+1, predicate, stt_fid, end_fid, frame_num - 1))

        before_len = min(30, stt_fid)
        after_len = min(30, frame_num - end_fid - 1)
        plt.figure(0)
        count = 0
        for fid in range(stt_fid-before_len, end_fid+after_len):

            if count == 100:
                break

            count += 1

            colors = colors0
            if fid < stt_fid or fid > end_fid:
                colors = [[1,1,1], [1,1,1]]
            sbj_box = objs[sbj_tid][fid]
            obj_box = objs[obj_tid][fid]
            sbj_cls = tid2cls[sbj_tid]
            obj_cls = tid2cls[obj_tid]
            frame_path = os.path.join(frame_root, '%06d.JPEG' % fid)
            plt.ion()
            plt.axis('off')
            show_boxes(frame_path, [sbj_box, obj_box], ['%s, %s' % (sbj_cls, predicate), obj_cls], colors)
            plt.pause(0.00001)
            plt.cla()

        plt.close()


if __name__ == '__main__':
    data_root = '../../../data/VidOR/Data/VID/val/'
    anno_root = '../../../data/VidOR/anno/val/'

    for pkg in os.listdir(data_root):
        pkg_path = os.path.join(data_root, pkg)
        for vid in os.listdir(pkg_path):
            # video_dir_path = os.path.join(pkg_path, vid)
            # anno_path = os.path.join(anno_root, pkg, vid+'.json')
            video_dir_path = os.path.join(data_root, '0004/11566980553')
            anno_path = os.path.join(anno_root, '0004/11566980553.json')
            show_video(video_dir_path, anno_path)