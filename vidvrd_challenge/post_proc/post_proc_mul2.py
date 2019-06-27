import os
import json
import copy
from supplement_mul import temporal_nms
from post_proc_mul1 import cal_iou, connect, extend_traj

ignore_nms_set = set()

ignore_nms_set.add('crab')
ignore_nms_set.add('bird')
ignore_nms_set.add('chicken')
ignore_nms_set.add('duck')
ignore_nms_set.add('penguin')
ignore_nms_set.add('fish')
ignore_nms_set.add('stingray')
ignore_nms_set.add('crocodile')
ignore_nms_set.add('snake')
ignore_nms_set.add('turtle')
ignore_nms_set.add('antelope')
ignore_nms_set.add('bear')
ignore_nms_set.add('camel')
ignore_nms_set.add('cat')
ignore_nms_set.add('cattle/cow')
ignore_nms_set.add('dog')
ignore_nms_set.add('elephant')
ignore_nms_set.add('hamster/rat')
ignore_nms_set.add('horse')
ignore_nms_set.add('kangaroo')
ignore_nms_set.add('leopard')
ignore_nms_set.add('lion')
ignore_nms_set.add('panda')
ignore_nms_set.add('pig')
ignore_nms_set.add('rabbit')
ignore_nms_set.add('sheep/goat')
ignore_nms_set.add('squirrel')
ignore_nms_set.add('tiger')
ignore_nms_set.add('adult')
ignore_nms_set.add('baby')
ignore_nms_set.add('child')


def split_trajectory_by_tracking(frame_dir, traj, vis=True):
    import matplotlib.pyplot as plt
    import cv2

    if vis:
        plt.figure(0)

    traj_splits = []
    tracker = None

    org_fids = sorted([int(fid) for fid in traj])
    org_stt_fid = org_fids[0]
    org_end_fid = org_fids[-1]

    print('-------------')
    print('%d -> %d' % (org_stt_fid, org_end_fid))

    need_init = False
    curr_split_stt_fid = org_stt_fid
    for fid in range(org_stt_fid, org_end_fid):
        frame_path = os.path.join(frame_dir, '%06d.JPEG' % fid)
        frame = cv2.imread(frame_path)
        im_h, im_w, _ = frame.shape

        curr_box = traj['%06d' % fid]
        if (fid - org_stt_fid) % 60 == 0 or need_init:
            # init tracker
            tracker = cv2.TrackerKCF_create()
            init_box = (curr_box[0],
                        curr_box[1],
                        curr_box[2] - curr_box[0] + 1,
                        curr_box[3] - curr_box[1] + 1)
            tracker.init(frame, init_box)
            box = curr_box
            next_box = None
            need_init = False
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

            next_box = traj['%06d' % (fid + 1)]
            if (not ok) or cal_iou(next_box, box) < 0.5 or fid == org_end_fid - 1:
                # generate a trajectory split
                print('split [%d %d]' % (curr_split_stt_fid, fid - 1))
                split_traj = {}
                for split_fid in range(curr_split_stt_fid, fid):
                    split_traj['%06d' % split_fid] = traj['%06d' % split_fid]
                traj_splits.append(split_traj)
                need_init = True
                curr_split_stt_fid = fid

        if vis:
            plt.ion()
            plt.axis('off')
            frame_show = plt.imread(frame_path)
            plt.imshow(frame_show)

            if box is not None:
                rect = plt.Rectangle((box[0], box[1]),
                                     box[2] - box[0],
                                     box[3] - box[1], fill=False,
                                     edgecolor=[0, 1, 0], linewidth=2)
                plt.gca().add_patch(rect)
            if next_box is not None:
                rect = plt.Rectangle((next_box[0], next_box[1]),
                                     next_box[2] - next_box[0],
                                     next_box[3] - next_box[1], fill=False,
                                     edgecolor=[1.0, 0, 0], linewidth=2)
                plt.gca().add_patch(rect)

            plt.show()
            plt.pause(0.0000001)
            plt.cla()

    if vis:
        plt.close()

    print('we get %d splits' % len(traj_splits))
    print('-------------')
    frame_list = sorted(os.listdir(frame_dir))

    # tracking 180 frame bidirectional
    for i, traj_split in enumerate(traj_splits):
        new_traj_split = extend_traj(traj_split, i, frame_list, frame_dir, max_new_box=180)
        traj_splits[i] = new_traj_split

    return traj_splits


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


def cal_vcover(det1, det2, cover_thr=0.7):
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


def special_nms(dets, vcover_thr=0.7):
    remove_ids = set()
    pairs = set()

    for i in range(len(dets)):
        for j in range(len(dets)):
            if i == j:
                continue
            else:
                vcover_ratio = cal_vcover(dets[i], dets[j])
                if vcover_ratio > vcover_thr:
                    if ('%s+%s' % (j, i)) not in pairs:
                        remove_ids.add(j)
                        pairs.add('%d+%d' % (i, j))

    last_dets = []
    for i in range(len(dets)):
        if i not in remove_ids:
            last_dets.append(dets[i])
    return last_dets


def tricky_check(dets, vid):
    det_num0 = len(dets)
    nms_dets = []

    all_cls_dets = {}
    for det in dets:
        if det['category'] in all_cls_dets:
            cls_dets = all_cls_dets[det['category']]
            cls_dets.append(det)
        else:
            cls_dets = [det]
            all_cls_dets[det['category']] = cls_dets

    det_num1 = 0
    det_num2 = 0
    for cls in all_cls_dets:
        cls_dets = all_cls_dets[cls]
        keep = temporal_nms(cls_dets)

        cls_dets_nms = []
        for i in keep:
            cls_dets_nms.append(cls_dets[i])
        det_num1 += len(cls_dets_nms)

        if cls not in ignore_nms_set:
            cls_dets_nms = special_nms(cls_dets_nms)
            det_num2 += len(cls_dets_nms)
        else:
            det_num2 += len(cls_dets_nms)
        nms_dets += cls_dets_nms

    print('%s [%d -> %d -> %d]' % (vid, det_num0, det_num1, det_num2))
    return nms_dets


def tracking_check(dets, vid, data_root):
    human_cls = set()
    human_cls.add('adult')

    checked_dets = []
    for i, det in enumerate(dets):

        if det['category'] not in human_cls:
            checked_dets.append(det)
        else:
            frame_dir = os.path.join(data_root, vid)
            traj_splits = split_trajectory_by_tracking(frame_dir, det['trajectory'])

            for traj_split in traj_splits:
                split_fids = sorted([int(fid) for fid in traj_split])

                if len(split_fids) < 0.1 * len(det['trajectory']):
                    continue

                det_split = dict()
                det_split['trajectory'] = traj_split
                det_split['start_fid'] = split_fids[0]
                det_split['end_fid'] = split_fids[-1]
                det_split['category'] = det['category']
                det_split['score'] = det['score']
                checked_dets.append(det_split)
    print('%s [%d -> %d]')
    return checked_dets


if __name__ == '__main__':
    split = 'val'
    res_path = '../evaluation/vidor_%s_object_pred_proc_all_2.json' % split
    data_root = '../../data/VidOR/Data/VID/'+split

    print('loading %s' % res_path)
    with open(res_path) as f:
        res = json.load(f)
        all_results = res['results']

    for vid in all_results:

        vid_dets = all_results[vid]
        vid_dets = tracking_check(vid_dets, vid, data_root)
        connect(vid_dets)
        vid_dets = tricky_check(vid_dets, vid)

        all_results[vid] = vid_dets

    sav_path = '../evaluation/vidor_%s_object_pred_proc_all_2_nms.json' % split
    print('saving %s' % sav_path)
    with open(sav_path, 'w') as f:
        res = {'results': all_results}
        json.dump(res, f)
