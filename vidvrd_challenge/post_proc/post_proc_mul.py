import os
import json
import math
import time
import numpy as np


cls_score_thr = {
    'bread': 0,
    'cake': 0.1,
    'dish': 0.2,
    'fruits': 0,
    'vegetables': 0.01,
    'backpack': 0.005,
    'camera': 0.01,
    'cellphone': 0.01,
    'handbag': 0.01,
    'laptop': 0.05,
    'suitcase': 0,
    'ball/sports_ball': 0.005,
    'bat': 0,
    'frisbee': 0,
    'racket': 0,
    'skateboard': 0,
    'ski': 0,
    'snowboard': 0,
    'surfboard': 0,
    'toy': 0.01,
    'baby_seat': 0.01,
    'bottle': 0.01,
    'chair': 0.01,
    'cup': 0.01,
    'electric_fan': 0.005,
    'faucet': 0.1,
    'microwave': 0,
    'oven': 0,
    'refrigerator': 0.01,
    'screen/monitor': 0.1,
    'sink': 0.1,
    'sofa': 0.1,
    'stool': 0.1,
    'table': 0.01,
    'toilet': 0.1,
    'guitar': 0.1,
    'piano': 0.4,
    'baby_walker': 0.01,
    'bench': 0.1,
    'stop_sign': 0,
    'traffic_light': 0,
    'aircraft': 0.005,
    'bicycle': 0.1,
    'bus/truck': 0.01,
    'car': 0.05,
    'motorcycle': 0.01,
    'scooter': 0,
    'train': 0,
    'watercraft': 0.1,
    'crab': 0,
    'bird': 0.2,
    'chicken': 0.2,
    'duck': 0.05,
    'penguin': 0.05,
    'fish': 0.1,
    'stingray': 0,
    'crocodile': 0,
    'snake': 0,
    'turtle': 0.4,
    'antelope': 0.01,
    'bear': 0,
    'camel': 0,
    'cat': 0.1,
    'cattle/cow': 0.01,
    'dog': 0.1,
    'elephant': 0.05,
    'hamster/rat': 0,
    'horse': 0.05,
    'kangaroo': 0.2,
    'leopard': 0.01,
    'lion': 0,
    'panda': 0,
    'pig': 0.1,
    'rabbit': 0.001,
    'sheep/goat': 0.01,
    'squirrel': 0,
    'tiger': 0,
    'adult': 0.1,
    'baby': 0.1,
    'child': 0.1
}

cls_dur_thr = {
    'bread': 1,
    'cake': 20,
    'dish': 20,
    'fruits': 1,
    'vegetables': 20,
    'backpack': 5,
    'camera': 5,
    'cellphone': 5,
    'handbag': 10,
    'laptop': 20,
    'suitcase': 1,
    'ball/sports_ball': 5,
    'bat': 1,
    'frisbee': 1,
    'racket': 1,
    'skateboard': 1,
    'ski': 1,
    'snowboard': 3,
    'surfboard': 3,
    'toy': 20,
    'baby_seat': 20,
    'bottle': 5,
    'chair': 20,
    'cup': 5,
    'electric_fan': 10,
    'faucet': 20,
    'microwave': 5,
    'oven': 1,
    'refrigerator': 20,
    'screen/monitor': 20,
    'sink': 20,
    'sofa': 20,
    'stool': 20,
    'table': 20,
    'toilet': 20,
    'guitar': 20,
    'piano': 20,
    'baby_walker': 20,
    'bench': 20,
    'stop_sign': 10,
    'traffic_light': 1,
    'aircraft': 5,
    'bicycle': 20,
    'bus/truck': 20,
    'car': 20,
    'motorcycle': 10,
    'scooter': 10,
    'train': 20,
    'watercraft': 10,
    'crab': 1,
    'bird': 20,
    'chicken': 5,
    'duck': 20,
    'penguin': 20,
    'fish': 20,
    'stingray': 5,
    'crocodile': 20,
    'snake': 1,
    'turtle': 20,
    'antelope': 10,
    'bear': 1,
    'camel': 10,
    'cat': 20,
    'cattle/cow': 20,
    'dog': 20,
    'elephant': 20,
    'hamster/rat': 5,
    'horse': 20,
    'kangaroo': 20,
    'leopard': 20,
    'lion': 1,
    'panda': 10,
    'pig': 20,
    'rabbit': 5,
    'sheep/goat': 20,
    'squirrel': 1,
    'tiger': 1,
    'adult': 20,
    'baby': 20,
    'child': 20
}


def cal_viou(det1, det2, iou_thr=0.7):
    stt_fid1 = int(det1['start_fid'])
    end_fid1 = int(det1['end_fid'])

    stt_fid2 = int(det2['start_fid'])
    end_fid2 = int(det2['end_fid'])

    inter_stt_fid = max(stt_fid1, stt_fid2)
    inter_end_fid = min(end_fid1, end_fid2)

    union_stt_fid = min(stt_fid1, stt_fid2)
    union_end_fid = max(end_fid1, end_fid2)

    overlap_frame_count = 0
    traj1 = det1['trajectory']
    traj2 = det2['trajectory']
    for fid in range(inter_stt_fid, inter_end_fid + 1):
        if cal_iou(traj1['%06d' % fid], traj2['%06d' % fid]) > iou_thr:
            overlap_frame_count += 1
    viou = overlap_frame_count * 1.0 / (union_end_fid - union_stt_fid + 1)
    return viou


def filler_bad_trajs(video_dets, max_per_vid=25):
    cands = []
    lasts = []
    for det in video_dets:
        det_dur = det['end_fid'] - det['start_fid'] + 1
        det_cls = det['category']
        det_scr = det['score']
        if det_dur >= cls_dur_thr[det_cls] and det_scr >= cls_score_thr[det_cls]:
            cands.append(det)
        else:
            lasts.append(det)

    cands = sorted(cands, key=lambda det: det['score'], reverse=True)
    cands = cands[:max_per_vid]
    if len(cands) < max_per_vid:
        lasts = sorted(lasts, key=lambda det: det['score'], reverse=True)
        lasts = lasts[: (max_per_vid - len(cands))]
        cands += lasts
    return cands


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
        i_area = i_w * i_h
        area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
        area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

        u_area = area1 + area2 - i_area
        iou = i_area * 1.0 / u_area
    return iou


def merge_traj(det1, det2):
    traj1 = det1['trajectory']
    traj2 = det2['trajectory']
    s1 = int(sorted(traj1.keys())[0])
    e1 = int(sorted(traj1.keys())[-1])
    s2 = int(sorted(traj2.keys())[0])
    e2 = int(sorted(traj2.keys())[-1])

    if e2 < s1 or s2 > e1:
        return None

    cnt = 0
    intersec_s = max(s1, s2)
    intersec_e = min(e1, e2)
    for i in range(intersec_s, intersec_e + 1):
        box1 = traj1['%06d' % i]
        box2 = traj2['%06d' % i]
        iou = cal_iou(box1, box2)
        if iou > 0.7:
            cnt += 1

    merged_det = None
    if cnt > (intersec_e - intersec_s + 1) * 0.7:
        # merge
        score1 = det1['score']
        score2 = det2['score']

        union_s = min(s1, s2)
        union_e = max(e1, e2)

        if score1 >= score2:
            for i in range(union_s, union_e + 1):
                if s1 <= i <= e1:
                    continue
                else:
                    traj1['%06d' % i] = traj2['%06d' % i]
            det1['start_fid'] = union_s
            det1['end_fid'] = union_e
            merged_det = det1
        else:
            for i in range(union_s, union_e + 1):
                if s2 <= i <= e2:
                    continue
                else:
                    traj2['%06d' % i] = traj1['%06d' % i]
            det2['start_fid'] = union_s
            det2['end_fid'] = union_e
            merged_det = det2

    return merged_det


def connect(video_dets):
    cont = True
    while cont:
        cont = False
        del_det_inds = set()
        new_dets = []
        for i in range(len(video_dets) - 1):
            # assume before
            det1 = video_dets[i]
            traj1 = det1['trajectory']
            cls1 = det1['category']
            s1 = int(sorted(traj1.keys())[0])
            e1 = int(sorted(traj1.keys())[-1])
            for j in range(i + 1, len(video_dets)):

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
                        cont = True
                        break
            if cont:
                break

        if cont:
            del_inds = sorted(list(del_det_inds), reverse=True)
            for ind in del_inds:
                video_dets.pop(ind)
            for det in new_dets:
                video_dets.append(det)


def track(frame_paths, init_box, vis=False, max_new_box_num=1000):
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
        im_h, im_w, _ = frame.shape
        if i == 0:
            # init tracker
            box = (init_box[0], init_box[1], init_box[2] - init_box[0], init_box[3] - init_box[1])
            status = tracker.init(frame, box)
            box = init_box
            if not status:
                break
        else:
            ok, box = tracker.update(frame)
            # [x1,y1,w,h] -> [x1,y1,x2,y2]
            box = [int(box[0]),
                   int(box[1]),
                   int(box[0] + box[2]),
                   int(box[1] + box[3])]
            box = [max(0, box[0]),
                   max(0, box[1]),
                   max(0, box[2]),
                   max(0, box[3])]
            box = [min(box[0], im_w-1),
                   min(box[1], im_h-1),
                   min(box[2], im_w-1),
                   min(box[3], im_h-1)]
            if (not ok) or is_over(box, im_w, im_h):
                break
            new_boxes.append(box)
            new_box_cnt += 1
            if new_box_cnt == max_new_box_num:
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
        if det_h * 1.0 / det_w > 3 or det_h < im_h * 0.1 or det_w < im_w * 0.1:
            return True
    if y1 < 5 or y2 > (im_h - 5):
        # disappear from top/bottom edge
        if det_w * 1.0 / det_h > 3 or det_h < im_h * 0.1 or det_w < im_w * 0.1:
            return True
    if det_w < im_w * 0.01 or det_h < im_h * 0.01 or max(det_w, det_h) * 1.0 / min(det_w, det_h) > 5:
        # disappear from center
        # occluded
        return True
    return False


def extend_traj(det, tid, frame_list, video_dir):
    w = det['width']
    h = det['height']
    cate = det['category']
    traj = det['trajectory']

    boxes = sorted(traj.items(), key=lambda d: int(d[0]))
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

    CACHE_LEN = 30
    if not head_is_over:
        # tracking backward
        # print('\t[%d] head track: <%s>' % (tid, cate))
        if traj_duration > 2 * CACHE_LEN:
            track_stt_fid = traj_stt_fid + CACHE_LEN
            curr_cache_len = CACHE_LEN
        else:
            track_stt_fid = int(round((traj_stt_fid + traj_end_fid) / 2.0))
            curr_cache_len = track_stt_fid - traj_stt_fid

        seg_frames = frame_list[track_stt_fid::-1]
        seg_frame_paths = [os.path.join(video_dir, frame_id) for frame_id in seg_frames]
        new_boxes = track(seg_frame_paths, traj['%06d' % track_stt_fid], vis=False)
        print('\t[%d] head add: %d <%s>' % (tid, len(new_boxes) - curr_cache_len, cate))

        for i in range(len(new_boxes)):
            new_box = new_boxes[i]
            frame_id = '%06d' % (int(track_stt_fid) - i - 1)
            traj[frame_id] = new_box

    if not tail_is_over:
        # tracking forward
        # print('\t[%d] tail track: <%s>' % (tid, cate))

        if traj_duration > 2 * CACHE_LEN:
            track_stt_fid = traj_end_fid - CACHE_LEN
            curr_cache_len = CACHE_LEN
        else:
            track_stt_fid = int(round((traj_stt_fid + traj_end_fid) / 2.0))
            curr_cache_len = traj_end_fid - track_stt_fid

        seg_frames = frame_list[track_stt_fid:]
        seg_frame_paths = [os.path.join(video_dir, frame_id) for frame_id in seg_frames]
        new_boxes = track(seg_frame_paths, traj['%06d' % track_stt_fid], vis=False)
        print('\t[%d] tail add: %d <%s>' % (tid, len(new_boxes) - curr_cache_len, cate))

        for i in range(len(new_boxes)):
            new_box = new_boxes[i]
            frame_id = '%06d' % (int(track_stt_fid) + i + 1)
            traj[frame_id] = new_box

    if head_is_over and tail_is_over:
        print('\t[%d] complete traj <%s>' % (tid, cate))

    boxes = sorted(traj.items(), key=lambda d: int(d[0]))
    det['start_fid'] = int(boxes[0][0])
    det['end_fid'] = int(boxes[-1][0])
    return det


def post_process(res_path, sav_path, data_root):
    from multiprocessing.pool import Pool as Pool
    from multiprocessing import cpu_count

    if os.path.exists(sav_path):
        print(sav_path + ' exists.')
        return

    # load predictions
    with open(res_path) as f:
        res = json.load(f)
        all_results = res['results']

    for v, video_id in enumerate(sorted(all_results)):

        t = time.time()
        print('=' * 30)
        print('[%d/%d] Proc %s' % (len(all_results), v + 1, video_id))
        video_dir = os.path.join(data_root, video_id)
        frame_list = sorted(os.listdir(video_dir))
        video_dets = all_results[video_id]
        org_det_num = len(video_dets)

        # maximum number of trajectories
        # max_per_vid = 20 + max((round(len(frame_list) / 900.0) - 1), 0) * 10
        # max_per_vid = min(max_per_vid, 40)
        max_per_vid = 20

        # filter out extremely short and low scored ones
        video_dets = filler_bad_trajs(video_dets, max_per_vid=int(max_per_vid))
        fil_det_num = len(video_dets)

        print('tracking: %d' % fil_det_num)

        # extend trajectories by tracking
        pool = Pool(processes=cpu_count())
        results = [pool.apply_async(extend_traj, args=(det, d, frame_list, video_dir))
                   for d, det in enumerate(video_dets)]
        pool.close()
        pool.join()
        for d in range(len(video_dets)):
            video_dets[d] = results[d].get()

        connect(video_dets)
        all_results[video_id] = video_dets
        t1 = time.time()
        print('\t%s Det num: %d -> %d -> %d (%.2f sec)' % (
        video_id, org_det_num, fil_det_num, len(video_dets), (t1 - t)))

    with open(sav_path, 'w') as f:
        json.dump(res, f)


def check_fid(res):
    for vid in res:
        print(vid)
        vid_dets = res[vid]
        for det in vid_dets:
            traj = det['trajectory']
            fids = sorted(int(fid) for fid in traj.keys())
            stt_fid = fids[0]
            end_fid = fids[-1]
            if det['start_fid'] != stt_fid or det['end_fid'] != end_fid:
                print('inconsistent !')
                det['start_fid'] = stt_fid
                det['end_fid'] = end_fid


if __name__ == '__main__':
    split = 'val'
    res_ids = [0, 1, 2, 3]

    data_root = '../../data/VidOR/Data/VID/%s' % split
    for res_id in res_ids:
        res_path = '../evaluation/vidor_%s_object_pred%d.json' % (split, res_id)
        sav_path = res_path[:-5] + '_proc.json'
        t = time.time()
        post_process(res_path, sav_path, data_root)
        t1 = time.time()
        dur = int(t1 - t)
        h = dur / 60 / 60
        m = dur / 60 - h * 60
        s = dur - h * 60 * 60 - m * 60
        print('Post process takes: %dh, %dm, %ds.' % (h, m, s))

    res_all = None
    for res_id in res_ids:
        res_path = '../evaluation/vidor_%s_object_pred%d.json' % (split, res_id)
        sav_path = res_path[:-5] + '_proc.json'
        with open(sav_path) as f:
            res = json.load(f)
            check_fid(res['results'])
        if res_all is None:
            res_all = res
        else:
            res_all['results'].update(res['results'])

    sav_path = '../evaluation/vidor_%s_object_pred_proc_all.json' % (split)
    with open(sav_path, 'w') as f:
        json.dump(res_all, f)
