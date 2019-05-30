import os
import json

import numpy as np


def cal_cover_ratio(det1, det2, iou_thr=0.7):

    stt_fid1 = det1['start_fid']
    end_fid1 = det1['end_fid']
    dur1 = end_fid1 - stt_fid1 + 1

    stt_fid2 = det2['start_fid']
    end_fid2 = det2['end_fid']
    dur2 = end_fid2 - stt_fid2 + 1

    if dur1 > dur2:
        long_det = det1
        short_det = det2
    else:
        long_det = det2
        short_det = det1

    short_stt_fid = short_det['start_fid']
    short_end_fid = short_det['end_fid']

    long_stt_fid = long_det['start_fid']
    long_end_fid = long_det['end_fid']

    inter_stt_fid = max(short_stt_fid, long_stt_fid)
    inter_end_fid = min(short_end_fid, long_end_fid)

    overlap_frame_count = 0
    traj1 = det1['trajectory']
    traj2 = det2['trajectory']
    for fid in range(inter_stt_fid, inter_end_fid+1):
        iou = cal_iou(traj1['%06d' % fid], traj2['%06d'% fid])
        print(iou)
        if iou > iou_thr:
            overlap_frame_count += 1
    cover_ratio = overlap_frame_count * 1.0 / (short_end_fid - short_stt_fid + 1)
    return cover_ratio


def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    i_xmin = max(xmin1, xmin2)
    i_ymin = max(ymin1, ymin2)
    i_xmax = min(xmax1, xmax2)
    i_ymax = min(ymax1, ymax2)
    i_w = i_xmax - i_xmin + 1
    i_h = i_ymax - i_ymin + 1
    iou = 0.0
    if i_w > 0 and i_h > 0:
        u_xmin = min(xmin1, xmin2)
        u_ymin = min(ymin1, ymin2)
        u_xmax = max(xmax1, xmax2)
        u_ymax = max(ymax1, ymax2)
        u_w = u_xmax - u_xmin + 1
        u_h = u_ymax - u_ymin + 1

        iou = (i_w * i_h) * 1.0 / (u_w * u_h)
    return iou


def gen_cover_mat(video_root, pred_path, vid=None):

    with open(pred_path) as f:
        pred_res = json.load(f)
        vid_res = pred_res['results']

    if vid is not None:
        vid_res = {vid: vid_res[vid]}

    for vid in vid_res:

        frame_dir = os.path.join(video_root, vid)
        frame_list = sorted(os.listdir(frame_dir))
        frame_num = len(frame_list)
        print('>>>> %s [%d] <<<<' % (vid, frame_num))

        video_dets = vid_res[vid]
        video_dets = sorted(video_dets, key=lambda det: det['score'])
        cover_mat = np.zeros((len(video_dets), len(video_dets)))

        for i in range(len(video_dets)-1):
            for j in range(i+1, len(video_dets)):
                det1 = video_dets[i]
                cls1 = det1['category']

                det2 = video_dets[j]
                cls2 = det2['category']

                if cls1 == cls2 == 'cup':
                    stt1 = det1['start_fid']
                    end1 = det1['end_fid']

                    stt2 = det2['start_fid']
                    end2 = det2['end_fid']

                    inter_stt = max(stt1, stt2)
                    inter_end = min(end1, end2)
                    if inter_end > inter_stt:
                        cover_ratio = cal_cover_ratio(det1, det2, 0.5)
                        cover_mat[i, j] = cover_ratio
                        cover_mat[j, i] = cover_ratio
                        print('%s: [%d -> %d] [%d -> %d] %.2f' % (cls1, stt1, end1, stt2, end2, cover_ratio))




if __name__ == '__main__':
    video_root = '../../data/VidOR/Data/VID/val'
    res_path = 'vidor_val_object_pred.json'
    vid = u'0004/11566980553'
    gen_cover_mat(video_root, res_path, vid)