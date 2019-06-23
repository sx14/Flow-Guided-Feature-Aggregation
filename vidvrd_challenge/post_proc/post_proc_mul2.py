import os
import json
from supplement_mul import temporal_nms

special_nms_cls = ['sofa', 'ball/sports_ball']


def cal_cover(box1, box2):
    # box2's cover ratio

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
        i_area = i_w * i_h
        area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
        iou = i_area * 1.0 / area2
    return iou


def cal_vcover(det1, det2, cover_thr=0.8):
    # det2's cover ratio

    traj1 = det1['trajectory']
    fids1 = sorted([int(fid) for fid in traj1.keys()])
    stt_fid1 = fids1[0]
    end_fid1 = fids1[-1]
    dur1 = end_fid1 - stt_fid1 + 1

    traj2 = det2['trajectory']
    fids2 = sorted([int(fid) for fid in traj2.keys()])
    stt_fid2 = fids2[0]
    end_fid2 = fids2[-1]
    dur2 = end_fid2 - stt_fid2 + 1

    inter_stt_fid = max(stt_fid1, stt_fid2)
    inter_end_fid = min(end_fid1, end_fid2)

    overlap_frame_count = 0
    traj1 = det1['trajectory']
    traj2 = det2['trajectory']
    for fid in range(inter_stt_fid, inter_end_fid+1):
        if cal_cover(traj1['%06d' % fid], traj2['%06d' % fid]) > cover_thr:
            overlap_frame_count += 1
    # viou = overlap_frame_count * 1.0 / (union_end_fid - union_stt_fid + 1)
    viou = overlap_frame_count * 1.0 / dur2
    return viou


def special_nms(dets, vcover_thr=0.8):
    remove_ids = set()
    for i in range(len(dets)):
        for j in range(len(dets)):
            if i == j:
                continue
            else:
                vcover_ratio = cal_vcover(dets[i], dets[j])
                if vcover_ratio > vcover_thr:
                    remove_ids.add(j)
    last_dets = []
    for i in range(len(dets)):
        if i not in remove_ids:
            last_dets.append(dets[i])
    return last_dets


def proc_video_detections(dets, vid):
    det_num0 = len(dets)
    nms_dets = []

    all_cls_dets = {}
    for det in dets:
        if det['category'] in all_cls_dets:
            cls_dets = all_cls_dets[det['category']]
            cls_dets.append(det)
        else:
            cls_dets = [det]
            all_cls_dets['category'] = cls_dets

    det_num1 = 0
    det_num2 = 0
    for cls in all_cls_dets:
        cls_dets = all_cls_dets[cls]
        keep = temporal_nms(cls_dets)

        cls_dets_nms = []
        for i in keep:
            cls_dets_nms.append(cls_dets[i])
        det_num1 += len(cls_dets_nms)

        if cls in special_nms_cls:
            cls_dets_nms = special_nms(cls_dets_nms)
            det_num2 += len(cls_dets_nms)
        else:
            det_num2 += len(cls_dets_nms)
        nms_dets += cls_dets_nms

    print('%s [%d -> %d -> %d]' % (vid, det_num0, det_num1, det_num2))
    return nms_dets


if __name__ == '__main__':
    split = 'val'
    res_path = '../evaluation/vidor_%s_object_pred_proc_all_2.json' % split
    with open(res_path) as f:
        res = json.load(f)
        all_results = res['results']

    for vid in all_results:
        vid_dets = all_results[vid]
        all_results[vid] = proc_video_detections(vid_dets)

    sav_path = '../evaluation/vidor_%s_object_pred_proc_all_2_nms.json' % split
    with open(sav_path, 'w') as f:
        res = {'results': all_results}
        json.dump(res, f)
