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

    stt_fids = np.array([int(det['start_fid']) for det in dets])
    end_fids = np.array([int(det['end_fid']) for det in dets])
    durs = end_fids - stt_fids + 1
    order = durs.argsort()[::-1]

    rm_ids = set()

    for i in range(len(order)-1):
        for j in range(i+1, len(order)):
            # len(i) >= len(j)
            det1 = dets[order[i]]
            det2 = dets[order[j]]

            stt_fid1 = det1['start_fid']
            end_fid1 = det1['end_fid']

            stt_fid2 = det2['start_fid']
            end_fid2 = det2['end_fid']

            inter_stt_fid = max(stt_fid1, stt_fid2)
            inter_end_fid = min(end_fid1, end_fid2)

            tiou = (inter_end_fid - inter_stt_fid + 1) * 1.0 / (end_fid2 - stt_fid2 + 1)

            if tiou > t_iou_thr:
                cov_ratio = cal_cover_ratio(det2, det1)
                if cov_ratio > 0.7 and (det1['score'] - det2['score'] > 0.2):
                    rm_ids.add(order[j])

    keep = [id for id in range(order) if id not in rm_ids]
    return keep


def filler_bad_trajs(video_dets, score_thr=0.1, min_len=5, max_per_vid=25):
    video_dets = sorted(video_dets, key=lambda det: det['score'], reverse=True)
    for i, det in enumerate(video_dets):
        if det['score'] < score_thr:
            break
    video_dets = video_dets[:i]
    video_dets = [det for det in video_dets if ((det['end_fid']-det['start_fid'] + 1) >= min_len)]
    video_dets = video_dets[:min(len(video_dets), max_per_vid)]
    return video_dets


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
    new_box_cnt = 0
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
            new_box_cnt += 1
            if new_box_cnt == 300:
                break
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
    det_w = x2 - x1 + 1
    det_h = y2 - y1 + 1

    # check position
    if x1 < 5 or x2 > (im_w - 5):
        # disappear from left/right edge
        if det_h * 1.0 / det_w > 3:
            return True
    if y1 < 5 or y2 > (im_h - 5):
        # disappear from top/bottom edge
        if det_w * 1.0 / det_h > 3:
            return True
    if det_w < 5 or det_h < 5 or max(det_w, det_h) * 1.0 / min(det_w, det_h) > 3:
        # disappear from center
        # occluded
        return True
    return False


def extend_traj(det, tid, frame_list, video_dir):
    w = det['width']
    h = det['height']
    cate = det['category']
    traj = det['trajectory']

    boxes = sorted(traj.items(), key=lambda d: d[0])
    traj_stt_fid = int(boxes[0][0])
    traj_end_fid = int(boxes[-1][0])
    traj_duration = traj_end_fid - traj_stt_fid + 1

    if traj_stt_fid == 0:
        head_is_over = True
    else:
        head_is_over = is_over(boxes[0][1], w, h)

    if traj_end_fid == (len(frame_list) - 1):
        tail_is_over = True
    else:
        tail_is_over = is_over(boxes[-1][1], w, h)

    cache_len = 30
    if not head_is_over:
        # tracking backward
        # print('\t[%d] head track: <%s>' % (tid, cate))
        if traj_duration > 2 * cache_len:
            track_stt_fid = traj_stt_fid + cache_len
        else:
            track_stt_fid = round((traj_stt_fid + traj_end_fid) / 2.0)
            cache_len = track_stt_fid - traj_stt_fid

        seg_frames = frame_list[track_stt_fid::-1]
        seg_frame_paths = [os.path.join(video_dir, frame_id) for frame_id in seg_frames]
        new_boxes = track(seg_frame_paths, traj['%06d' % track_stt_fid], vis=False)
        print('\t[%d] head add: %d <%s>' % (tid, len(new_boxes) - cache_len, cate))

        for i in range(len(new_boxes)):
            frame_id = '%06d' % (int(track_stt_fid) - i - 1)
            new_box = new_boxes[i]
            traj[frame_id] = new_box

    if not tail_is_over:
        # tracking forward
        # print('\t[%d] tail track: <%s>' % (tid, cate))

        if traj_duration > 2 * cache_len:
            track_stt_fid = traj_end_fid - cache_len
        else:
            track_stt_fid = round((traj_stt_fid + traj_end_fid) / 2.0)
            cache_len = traj_end_fid - track_stt_fid

        seg_frames = frame_list[track_stt_fid:]
        seg_frame_paths = [os.path.join(video_dir, frame_id) for frame_id in seg_frames]
        new_boxes = track(seg_frame_paths, traj['%06d' % track_stt_fid], vis=False)
        print('\t[%d] tail add: %d <%s>' % (tid, len(new_boxes) - cache_len, cate))

        for i in range(len(new_boxes)):
            frame_id = '%06d' % (int(track_stt_fid) + i + 1)
            new_box = new_boxes[i]
            traj[frame_id] = new_box

    if head_is_over and tail_is_over:
        print('\t[%d] complete traj <%s>' % (tid, cate))

    boxes = sorted(traj.items(), key=lambda d: d[0])
    det['start_fid'] = int(boxes[0][0])
    det['end_fid'] = int(boxes[-1][0])


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

        max_per_vid = round(len(frame_list) / 1000.0)*25
        video_dets = filler_bad_trajs(video_dets, max_per_vid=int(max_per_vid))
        fillered_dets = len(video_dets)

        from multiprocessing.pool import ThreadPool as Pool
        pool = Pool()
        for d, det in enumerate(video_dets):
            pool.apply_async(extend_traj, args=(det, d, frame_list, video_dir))
        pool.close()
        pool.join()
        all_results[video_id] = video_dets
        print('\tDet num: %d -> %d -> %d' % (org_det_num, fillered_dets, len(video_dets)))
        # connect(video_dets)

    res_path1 = res_path[:-5] + '_proc.json'
    with open(res_path1, 'w') as f:
        json.dump(res, f)



res_path = '../evaluation/vidor_val_object_pred.json'
data_root = '../../data/VidOR/Data/VID/val'
post_process(res_path, data_root)