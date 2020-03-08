import os
import json
import xml.etree.ElementTree as ET


def is_spatial(predicate):
    spatial_set = set()
    spatial_set.add('towards')
    spatial_set.add('next_to')
    spatial_set.add('inside')
    spatial_set.add('in_front_of')
    spatial_set.add('beneath')
    spatial_set.add('behind')
    spatial_set.add('away')
    spatial_set.add('above')
    return predicate in spatial_set


def gen_vidor_rlt_obj_gt(video_anno_root, video_list, save_file_name):
    splits = ['validation, training']
    vid_val_gt = {}
    for i, vid in enumerate(video_list):
        # for each video
        print('Gen [%d/%d]' % (len(video_list), i+1))

        for split in splits:
            vid = vid[4:]
            vid_anno_path = os.path.join(video_anno_root, split, vid+'.json')
            if os.path.exists(vid_anno_path):
                break
        vid_anno = json.load(open(vid_anno_path))
        vid_frame_n = vid_anno['frame_count']
        vid_obj_clss = vid_anno['subject/objects']

        objs = dict()
        tid2cls = dict()
        for obj_cls in vid_obj_clss:
            tid2cls[obj_cls['tid']] = obj_cls['category']
            objs[obj_cls['tid']] = [None for _ in range(vid_frame_n)]

        # object complete trajs
        frame_boxes = vid_anno['trajectories']
        for f in range(len(frame_boxes)):
            boxes = frame_boxes[f]
            for box in boxes:
                tid = box['tid']
                xmin = box['bbox']['xmin']
                ymin = box['bbox']['ymin']
                xmax = box['bbox']['xmax']
                ymax = box['bbox']['ymax']
                objs[tid][f] = [xmin, ymin, xmax, ymax]

        # object segments in relations
        rlt_obj_tid = 0
        rlt_obj_trajs = []
        for rlt in vid_anno['relation_instances']:
            predicate = rlt['predicate']
            if not is_spatial(predicate):
                continue

            stt_fid = rlt['begin_fid']
            end_fid = rlt['end_fid']
            obj_tid = rlt['object_tid']
            sbj_tid = rlt['subject_tid']

            sbj_obj_tids = [sbj_tid, obj_tid]
            for tid in sbj_obj_tids:
                rlt_obj_traj = {}
                obj = objs[tid]
                for fid in range(stt_fid, end_fid):
                    rlt_obj_traj['%06d' % fid] = obj[fid]

                rlt_obj = {
                    'tid': rlt_obj_tid,
                    'category': tid2cls[tid],
                    'trajectory': rlt_obj_traj}
                rlt_obj_trajs.append(rlt_obj)
                rlt_obj_tid += 1

        vid_val_gt['/'.join(vid.split('/')[1:])] = rlt_obj_trajs

    with open(save_file_name, 'w') as f:
        json.dump(vid_val_gt, f)


if __name__ == '__main__':
    dataset_name = 'VidOR-HOID-mini'
    dataset_name1 = 'vidor_hoid_mini'
    vid_anno_root = '../../data/%s/anno' % dataset_name
    vid_list_path = '../../data/%s/ImageSets/VID_val_videos.txt' % dataset_name
    with open(vid_list_path) as f:
        video_list = [l.strip().split(' ')[0] for l in f.readlines()]
    save_path = '../evaluation/%sr_val_object_segment_gt.json' % dataset_name1
    gen_vidor_rlt_obj_gt(vid_anno_root, video_list, save_path)
