import os
import json

import numpy as np
import matplotlib.pyplot as plt

from post_proc_mul import track, connect, cal_viou
from vidvrd_challenge.evaluation.show_prediction import show_boxes


def temporal_nms(dets, tiou_thr=0.7):

    if len(dets) == 0:
        return []

    scores = np.array([det['score'] for det in dets])
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        keep_det = dets[i]

        vious = np.ones(len(order))
        for j in range(1, len(order)):
            viou = cal_viou(keep_det, dets[order[j]])
            vious[j] = viou

        inds = np.where(vious <= tiou_thr)[0]
        order = order[inds]

    return keep


def save_trajectory_detections(res_path, results):
    output = {
        "version": "VERSION 1.0",
        "results": results
    }
    with open(res_path, 'w') as f:
        json.dump(output, f)


def load_trajectory_detections(res_path):
    with open(res_path) as f:
        res = json.load(f)
        traj_dets = res['results']
    print('trajectory detection results loaded.')
    return traj_dets


def load_frame_detections(res_path):
    with open(res_path) as f:
        frame_dets = json.load(f)
    print('frame detection results loaded.')
    return frame_dets


def cal_ious(box, boxes):
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes).astype(np.float)
    xmins = boxes_np[:, 0]
    ymins = boxes_np[:, 1]
    xmaxs = boxes_np[:, 2]
    ymaxs = boxes_np[:, 3]
    areas = (xmaxs - xmins + 1) * (ymaxs - ymins + 1)

    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]
    area = (xmax - xmin) * (ymax - ymin)

    i_xmins = np.maximum(xmins, xmin)
    i_ymins = np.maximum(ymins, ymin)
    i_xmaxs = np.minimum(xmaxs, xmax)
    i_ymaxs = np.minimum(ymaxs, ymax)
    i_ws = i_xmaxs - i_xmins
    i_ws[i_ws < 0] = 0
    i_hs = i_ymaxs - i_ymins
    i_hs[i_hs < 0] = 0
    i_areas = i_ws * i_hs

    u_areas = areas + area - i_areas

    ious = i_areas / u_areas
    return ious.tolist()


def supplement_frame_detections(all_traj_dets, all_frame_dets, data_root, max_per_frame=5, max_per_video=20):

    print('supplement frame detections collecting ...')

    all_vids = sorted(all_frame_dets.keys())
    all_sup_frame_dets = {}
    for v, vid in enumerate(all_vids):
        # for each video
        vid_traj_dets = all_traj_dets[vid]
        vid_frame_dets = all_frame_dets[vid]

        vid_sup_frame_dets = []

        for fid in vid_frame_dets:

            # for each frame
            frame_dets = [det for det in vid_frame_dets[fid]]
            frame_dets = sorted(frame_dets, key=lambda det: det['score'], reverse=True)[:max_per_frame]

            # ==== show ====
            frame_path = os.path.join(data_root, vid, fid+'.JPEG')
            frame = plt.imread(frame_path)
            boxes = [det['box'] for det in frame_dets]
            confs = [det['score'] for det in frame_dets]
            cates = [det['category'] for det in frame_dets]
            # ==== show ====


            traj_boxes = []
            traj_clses = []
            traj_confs = []
            for traj_det in vid_traj_dets:
                if fid in traj_det['trajectory']:
                    traj_boxes.append(traj_det['trajectory'][fid])
                    traj_clses.append('>>>> %s <<<<' % traj_det['category'])
                    traj_confs.append(traj_det['score'])

            boxes += traj_boxes
            confs += traj_confs
            cates += traj_clses

            # show_boxes(frame, boxes, cates, confs, 'mul')

            for i, frame_det in enumerate(frame_dets):
                frame_det['fid'] = fid

                ious = cal_ious(frame_det['box'], traj_boxes)
                large_overlap = False
                inconsistent_cls = True
                for j in range(len(ious)):
                    if ious[j] > 0.5:
                        large_overlap = True
                        if traj_clses[j] == frame_det['category']:
                            inconsistent_cls = False

                if large_overlap:
                    if inconsistent_cls:
                        vid_sup_frame_dets.append(frame_det)
                else:
                    vid_sup_frame_dets.append(frame_det)

        vid_sup_frame_dets = sorted(vid_sup_frame_dets, key=lambda det: det['score'])[:max_per_video]
        all_sup_frame_dets[vid] = vid_sup_frame_dets
    return all_sup_frame_dets


def supplement_trajectories(all_traj_dets, sup_frame_dets, data_root, max_per_video=30):

    from multiprocessing.pool import Pool as Pool
    from multiprocessing import cpu_count

    all_vids = sorted(sup_frame_dets.keys())
    vid_num = len(all_vids)
    for v, vid in enumerate(all_vids):
        # for each video
        print('[%d/%d] supplement: %s' % (vid_num, v+1, vid))
        vid_traj_dets = all_traj_dets[vid]
        vid_frame_dets = sup_frame_dets[vid]
        vid_frame_root = os.path.join(data_root, vid)
        vid_frame_num = len(os.listdir(vid_frame_root))

        sup_trajs = []
        for d, det in enumerate(vid_frame_dets):
            sup_traj = det2traj(det, vid_frame_num, vid_frame_root)
            sup_trajs.append(sup_traj)

        # pool = Pool(processes=cpu_count())
        # results = [pool.apply_async(det2traj, args=(det, vid_frame_num, vid_frame_root))
        #            for d, det in enumerate(vid_frame_dets)]
        # pool.close()
        # pool.join()
        # sup_trajs = [result.get() for result in results]

        vid_traj_dets += sup_trajs
        vid_traj_det_num_org = len(vid_traj_dets)
        connect(vid_traj_dets)
        vid_traj_det_num_cnt = len(vid_traj_dets)

        all_cls_dets = {}
        for traj_det in vid_traj_dets:
            cls = traj_det['category']
            if cls in all_cls_dets:
                cls_dets = all_cls_dets[cls]
                cls_dets.append(traj_det)
            else:
                cls_dets = [traj_det]
                all_cls_dets[cls] = cls_dets

        nms_vid_traj_dets = []
        for cls in all_cls_dets:
            cls_dets = all_cls_dets[cls]
            keep = temporal_nms(cls_dets, 0.7)
            nms_cls_dets = [cls_dets[i] for i in keep]
            nms_vid_traj_dets += nms_cls_dets

        nms_vid_traj_dets = sorted(nms_vid_traj_dets, key=lambda det: det['score'], reverse=True)
        vid_traj_det_num_nms = len(nms_vid_traj_dets)
        print('%d -> %d -> %d' % (vid_traj_det_num_org, vid_traj_det_num_cnt, vid_traj_det_num_nms))

        all_traj_dets[vid] = nms_vid_traj_dets[:max_per_video]


def det2traj(det, frame_num, frame_root):
    fid = det['fid']
    box = det['box']
    cls = det['category']
    fid_int = int(fid)
    forward_frame_paths = [os.path.join(frame_root, '%06d.JPEG' % f) for f in range(fid_int + 1, frame_num)]
    backward_frame_paths = [os.path.join(frame_root, '%06d.JPEG' % f) for f in range(fid_int - 1, -1, -1)]
    forward_traj = track(forward_frame_paths, box, max_new_box_num=10000)
    backward_fraj = track(backward_frame_paths, box, max_new_box_num=10000)
    traj = {'%06d' % fid_int: box}
    for i in range(len(forward_traj)):
        box = forward_traj[i]
        traj['%06d' % (fid_int + i + 1)] = box
    for i in range(len(backward_fraj)):
        box = backward_fraj[i]
        traj['%06d' % (fid_int - i - 1)] = box

    traj_fids = sorted([int(fid) for fid in traj])
    traj_det = {
        'category': det['category'],
        'score': det['score'],
        'trajectory': traj,
        'start_fid': '%06d' % traj_fids[0],
        'end_fid': '%06d' % traj_fids[-1]
    }

    print('\t[ %d | %d | %d ] %s(%d) ' % (traj_fids[0], fid_int, traj_fids[-1], cls, len(traj_fids)))

    return traj_det


if __name__ == '__main__':
    split = 'val'

    traj_det_path = os.path.abspath('../evaluation/vidor_%s_object_pred_proc_all.json' % split)
    frame_det_path = os.path.abspath('vidor_%s_object_pred_frame.json' % split)
    data_root = os.path.abspath('../../data/VidOR/Data/VID/%s' % split)

    all_traj_dets = load_trajectory_detections(traj_det_path)
    all_frame_dets = load_frame_detections(frame_det_path)

    sup_frame_dets = supplement_frame_detections(all_traj_dets, all_frame_dets, data_root)
    supplement_trajectories(all_traj_dets, sup_frame_dets, data_root)

    output_path = os.path.abspath('../evaluation/vidor_%s_object_pred_proc_all_sup.json' % split)
    save_trajectory_detections(output_path, all_traj_dets)

