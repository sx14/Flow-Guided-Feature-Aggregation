import os
import json

import numpy as np


def cal_viou(det1, det2, iou_thr=0.7):

    stt_fid1 = det1['start_fid']
    end_fid1 = det1['end_fid']

    stt_fid2 = det2['start_fid']
    end_fid2 = det2['end_fid']

    inter_stt_fid = max(stt_fid1, stt_fid2)
    inter_end_fid = min(end_fid1, end_fid2)

    union_stt_fid = min(stt_fid1, stt_fid2)
    union_end_fid = max(end_fid1, end_fid2)

    overlap_frame_count = 0
    traj1 = det1['trajectory']
    traj2 = det2['trajectory']
    for fid in range(inter_stt_fid, inter_end_fid+1):
        if cal_iou(traj1['%06d' % fid], traj2['%06d'% fid]) > iou_thr:
            overlap_frame_count += 1
    viou = overlap_frame_count * 1.0 / (union_end_fid - union_stt_fid + 1)
    return viou


def cal_cover_ratio(short_det, long_det, iou_thr=0.7):

    short_stt_fid = short_det['start_fid']
    short_end_fid = short_det['end_fid']

    long_stt_fid = long_det['start_fid']
    long_end_fid = long_det['end_fid']

    inter_stt_fid = max(short_stt_fid, long_stt_fid)
    inter_end_fid = min(short_end_fid, long_end_fid)

    overlap_frame_count = 0
    traj1 = short_det['trajectory']
    traj2 = long_det['trajectory']
    for fid in range(inter_stt_fid, inter_end_fid+1):
        iou = cal_iou(traj1['%06d' % fid], traj2['%06d'% fid])
        # print(iou)
        if iou > iou_thr:
            overlap_frame_count += 1
    cover_ratio = overlap_frame_count * 1.0 / (long_end_fid - long_stt_fid + 1)
    return cover_ratio


def remove_covered(dets, cls):
    rm_inds = set()
    for i in range(len(dets)-1):
        for j in range(i+1, len(dets)):
            stt_fid1 = dets[i]['start_fid']
            end_fid1 = dets[i]['end_fid']
            stt_fid2 = dets[j]['start_fid']
            end_fid2 = dets[j]['end_fid']

            if stt_fid1 >= stt_fid2 and end_fid1 <= end_fid2:
                # i is covered
                cover_ratio = cal_cover_ratio(dets[i], dets[j])
                if cover_ratio > 0.7:
                    rm_inds.add(i)
            elif stt_fid2 >= stt_fid1 and end_fid2 <= end_fid1:
                # j is covered
                cover_ratio = cal_cover_ratio(dets[j], dets[i])
                # print('\t%.2f' % cover_ratio)
                if cover_ratio > 0.7:
                    rm_inds.add(i)

    rm_inds = sorted(list(rm_inds), reverse=True)
    # print('\t remove: %s %d' % (cls, len(rm_inds)))
    for i in rm_inds:
        dets.pop(i)


def temperal_nms(dets, cls, t_iou_thr=0.7):

    remove_covered(dets, cls)
    scores = np.array([det['score'] for det in dets])
    order = scores.argsort()[::-1]

    stt_fids = np.array([int(det['start_fid']) for det in dets])
    end_fids = np.array([int(det['end_fid']) for det in dets])

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        stt_fid1 = int(dets[i]['start_fid'])
        end_fid1 = int(dets[i]['end_fid'])

        inter_stt_fids = np.maximum(stt_fid1, stt_fids[order[1:]])
        inter_end_fids = np.minimum(end_fid1, end_fids[order[1:]])

        union_stt_fids = np.minimum(stt_fid1, stt_fids[order[1:]])
        union_end_fids = np.maximum(end_fid1, end_fids[order[1:]])

        t_iou = (inter_end_fids - inter_stt_fids + 1) * 1.0 / (union_end_fids - union_stt_fids + 1)

        for j in range(len(t_iou)):
            if t_iou[j] > t_iou_thr:
                t_iou[j] = cal_viou(dets[i], dets[j])

        inds = np.where(t_iou <= t_iou_thr)[0]
        order = order[inds + 1]
    return keep


def filler_short_trajs(video_dets, len_thr=5, score_thr=0.05):
    rm_inds = []
    for i, det in enumerate(video_dets):
        det_len = len(det['trajectory'].keys())
        det_scr = det['score']
        if det_len < len_thr or det_scr < score_thr:
            rm_inds.append(i)
    rm_inds.reverse()
    for i in rm_inds:
        video_dets.pop(i)
    return video_dets


def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    i_xmin = max(xmin1, xmin2)
    i_ymin = max(ymin1, ymin2)
    i_xmax = min(xmax1, xmax2)
    i_ymax = min(ymax1, ymax2)
    i_w = i_xmax - i_xmin
    i_h = i_ymax - i_ymin
    iou = 0.0
    if i_w > 0 and i_h > 0:
        u_xmin = min(xmin1, xmin2)
        u_ymin = min(ymin1, ymin2)
        u_xmax = max(xmax1, xmax2)
        u_ymax = max(ymax1, ymax2)
        u_w = u_xmax - u_xmin
        u_h = u_ymax - u_ymin

        iou = (i_w * i_h) * 1.0 / (u_w * u_h)
    return iou


def merge_traj(det1, det2):

    traj1 = det1['trajectory']
    traj2 = det2['trajectory']
    s1 = int(sorted(traj1.keys())[0])
    e1 = int(sorted(traj1.keys())[-1])
    s2 = int(sorted(traj2.keys())[0])
    e2 = int(sorted(traj2.keys())[-1])

    cnt = 0
    intersec_s = max(s1, s2)
    intersec_e = min(e1, e2)
    for i in range(intersec_s, intersec_e+1):
        box1 = traj1['%06d' % i]
        box2 = traj2['%06d' % i]
        iou = cal_iou(box1, box2)
        print(iou)

        if iou > 0.6:
            cnt += 1

    merged_det = None
    if cnt > (e1 - s2 + 1) * 0.6:
        # merge
        score1 = det1['score']
        score2 = det2['score']
        if score1 >= score2:
            for i in range(e1+1, e2+1):
                traj1['%6d' % i] = traj2['%6d' % i]
            merged_det = det1
        else:
            for i in range(s1, s2):
                traj2['%6d' % i] = traj1['%6d' % i]
            merged_det = det2

    return merged_det


def connect(video_dets):

    cont = True
    while cont:
        cont = False
        del_det_inds = set()
        new_dets = []
        for i in range(len(video_dets)-1):
            # assume before
            det1 = video_dets[i]
            traj1 = det1['trajectory']
            cls1 = det1['category']
            s1 = int(sorted(traj1.keys())[0])
            e1 = int(sorted(traj1.keys())[-1])
            for j in range(i+1, len(video_dets)):

                det2 = video_dets[j]
                traj2 = det2['trajectory']
                cls2 = det2['category']
                s2 = int(sorted(traj2.keys())[0])
                e2 = int(sorted(traj2.keys())[-1])

                if cls1 == cls2 and not (e2 < s1 or e1 < s2):
                    # merge
                    merged_det = merge_traj(det1, det2)
                    if merged_det is not None:
                        del_det_inds.add(i)
                        del_det_inds.add(j)
                        new_dets.append(merged_det)

        if len(new_dets) > 0:
            cont = True
            del_inds = sorted(list(del_det_inds), reverse=True)
            for ind in del_inds:
                video_dets.pop(ind)
            for det in new_dets:
                video_dets.append(det)


def track(frame_paths, init_box, vis=False):
    # init box: [x1, y1, x2, y2]
    import matplotlib.pyplot as plt
    import cv2

    if vis:
        plt.figure(0)
    new_boxes = []
    tracker = cv2.TrackerKCF_create()
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if i == 0:
            # init tracker
            im_h, im_w, _ = frame.shape
            box = (init_box[0], init_box[1], init_box[2]-init_box[0], init_box[3]-init_box[1])
            status = tracker.init(frame, box)
            box = init_box
            if not status:
                break
        else:
            ok, box = tracker.update(frame)
            box = [int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3])]
            if (not ok) or is_over(box, im_w, im_h):
                break
            new_boxes.append(box)
        if vis:
            plt.ion()
            plt.axis('off')
            # print('\t'+frame_path.split('/')[-1])
            frame_show = plt.imread(frame_path)
            plt.imshow(frame_show)

            if box is not None:
                rect = plt.Rectangle((box[0], box[1]),
                                     box[2] - box[0],
                                     box[3] - box[1], fill=False,
                                     edgecolor=[1.0, 0, 0], linewidth=2)
                plt.gca().add_patch(rect)
            plt.show()
            plt.pause(0.01)
            plt.cla()
    if vis:
        plt.close()
    return new_boxes


def is_over(det, im_w, im_h):
    x1, y1, x2, y2 = det
    det_w = x2 - x1
    det_h = y2 - y1

    # check position
    if x1 < 5 or x2 > (im_w - 5):
        # disappear from left/right edge
        if det_h * 1.0 / det_w > 3:
            return True
    if y1 < 5 or y2 > (im_h - 5):
        # disappear from top/bottom edge
        if det_w * 1.0 / det_h > 3:
            return True
    if det_w < 5 and det_h < 5:
        # disappear from center
        return True
    return False


def post_process(res_path, data_root):
    with open(res_path) as f:
        res = json.load(f)
        all_results = res['results']

    for v, video_id in enumerate(sorted(all_results)):

        print('=' * 30)
        print('[%d/%d] Proc %s' % (len(all_results), v+1, video_id))
        video_dir = os.path.join(data_root, video_id)
        frame_list = sorted(os.listdir(video_dir))
        video_dets = all_results[video_id]
        org_det_num = len(video_dets)

        filler_short_trajs(video_dets)
        fillered_dets = len(video_dets)

        cls_dets = {}
        for det in video_dets:
            det_cls = det['category']
            if det_cls not in cls_dets:
                cls_dets[det_cls] = [det]
            else:
                cls_dets[det_cls].append(det)

        video_dets = []
        for cls in cls_dets:
            dets = cls_dets[cls]
            keep = temperal_nms(dets, cls)
            dets = [dets[i] for i in keep]

            for d, det in enumerate(dets):

                w = det['width']
                h = det['height']
                cate = det['category']
                traj = det['trajectory']

                boxes = sorted(traj.items(), key=lambda d: d[0])
                org_start_fid = int(boxes[0][0])
                org_end_fid = int(boxes[-1][0])

                if org_start_fid == 0:
                    head_is_over = True
                else:
                    head_is_over = is_over(boxes[0][1], w, h)

                if org_end_fid == (len(frame_list) - 1):
                    tail_is_over = True
                else:
                    tail_is_over = is_over(boxes[-1][1], w, h)

                if not head_is_over:
                    # tracking backward
                    print('\t[%d] head track: <%s>' % (d, cate))
                    start_frame_id = int(boxes[0][0])
                    seg_frames = frame_list[start_frame_id::-1]
                    seg_frame_paths = [os.path.join(video_dir, frame_id) for frame_id in seg_frames]
                    new_boxes = track(seg_frame_paths, boxes[0][1], vis=False)
                    # print('\t[%d] head add: %d <%s>' % (d, len(new_boxes), cate))

                    for i in range(len(new_boxes)):
                        frame_id = '%06d' % (int(start_frame_id) - i - 1)
                        new_box = new_boxes[i]
                        traj[frame_id] = new_box

                if not tail_is_over:
                    # tracking forward
                    print('\t[%d] tail track: <%s>' % (d, cate))
                    start_frame_id = int(boxes[-1][0])
                    seg_frames = frame_list[start_frame_id:]
                    seg_frame_paths = [os.path.join(video_dir, frame_id) for frame_id in seg_frames]
                    new_boxes = track(seg_frame_paths, boxes[-1][1], vis=False)
                    # print('\t[%d] tail add: %d <%s>' % (d, len(new_boxes), cate))

                    for i in range(len(new_boxes)):
                        frame_id = '%06d' % (int(start_frame_id) + i + 1)
                        new_box = new_boxes[i]
                        traj[frame_id] = new_box

                if head_is_over and tail_is_over:
                    print('\t[%d] complete traj <%s>' % (d, cate))

                boxes = sorted(traj.items(), key=lambda d: d[0])
                det['start_fid'] = int(boxes[0][0])
                det['end_fid'] = int(boxes[-1][0])

            video_dets += dets

        all_results[video_id] = video_dets
        print('\tDet num: %d -> %d -> %d' % (org_det_num, fillered_dets, len(video_dets)))




        connect(video_dets)

    res_path1 = res_path[:-5] + '_proc.json'
    with open(res_path1, 'w') as f:
        json.dump(res, f)


# test tracker
# with open('../evaluation/imagenet_val_object_pred.json') as f:
#     all_res = json.load(f)
#     all_res = all_res['results']
#
# data_root = '../../data/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000001'
# traj = all_res['ILSVRC2015_val_00000001'][2]['trajectory']
# init_box = traj[sorted(traj.keys())[0]]
# frame_paths = [os.path.join(data_root, fid+'.JPEG') for fid in sorted(traj.keys())]
# track(frame_paths, init_box, traj)

res_path = '../evaluation/vidor_val_object_pred.json'
data_root = '../../data/VidOR/Data/VID/val'
post_process(res_path, data_root)