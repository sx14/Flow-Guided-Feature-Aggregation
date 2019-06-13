import os
import json
import numpy as np

from post_proc_mul import track, connect


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


def cal_iou(box, boxes):
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes)
    xmins = boxes_np[:, 0]
    ymins = boxes_np[:, 1]
    xmaxs = boxes_np[:, 2]
    ymaxs = boxes_np[:, 3]

    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]

    i_xmins = np.maximum(xmins, xmin)
    i_ymins = np.maximum(ymins, ymin)
    i_xmaxs = np.minimum(xmaxs, xmax)
    i_ymaxs = np.minimum(ymaxs, ymax)
    i_ws = i_xmaxs - i_xmins
    i_ws[i_ws < 0] = 0
    i_hs = i_ymaxs - i_ymins
    i_hs[i_hs < 0] = 0
    i_areas = i_ws * i_hs
    i_areas[i_areas < 0] = 0

    u_xmins = np.minimum(xmins, xmin)
    u_ymins = np.minimum(ymins, ymin)
    u_xmaxs = np.maximum(xmaxs, xmax)
    u_ymaxs = np.maximum(ymaxs, ymax)
    u_areas = (u_xmaxs - u_xmins) * (u_ymaxs - u_ymins)

    ious = i_areas / u_areas
    return ious.tolist()


def supplement_frame_detections(all_traj_dets, all_frame_dets, max_per_frame=100, max_per_video=10):

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

            traj_boxes = []
            traj_clses = []
            for traj_det in vid_traj_dets:
                if fid in traj_det['trajectory']:
                    traj_boxes.append(traj_det['trajectory'][fid])
                    traj_clses.append(traj_det['category'])

            for i, frame_det in enumerate(frame_dets):
                frame_det['fid'] = fid

                ious = cal_iou(frame_det['box'], traj_boxes)
                large_overlap = False
                inconsistent_cls = True
                for j in range(len(ious)):
                    if ious[j] > 0.7:
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


def supplement_trajectories(all_traj_dets, sup_frame_dets, data_root):

    from multiprocessing.pool import Pool as Pool
    from multiprocessing import cpu_count
    pool = Pool(processes=cpu_count())

    all_vids = sorted(sup_frame_dets.keys())
    vid_num = len(all_vids)
    for v, vid in enumerate(all_vids):
        # for each video
        print('[%d/%d] supplement: %s' % (vid_num, v+1, vid))
        vid_traj_dets = all_traj_dets[vid]
        vid_frame_dets = sup_frame_dets[vid]
        vid_frame_root = os.path.join(data_root, vid)
        vid_frame_num = len(os.listdir(vid_frame_root))

        results = [pool.apply_async(det2traj, args=(det, vid_frame_num, vid_frame_root))
                   for d, det in enumerate(vid_frame_dets)]
        pool.close()
        pool.join()

        sup_trajs = [result.get() for result in results]
        vid_traj_dets += sup_trajs

        connect(vid_traj_dets)


def det2traj(det, frame_num, frame_root):
    fid = det['fid']
    box = det['box']
    fid_int = int(fid)
    forward_frame_paths = [os.path.join(frame_root, '%06d.JPEG' % f) for f in range(fid_int + 1, frame_num)]
    backward_frame_paths = [os.path.join(frame_root, '%06d.JPEG' % f) for f in range(fid_int - 1, -1, -1)]
    forward_traj = track(box, forward_frame_paths, max_new_box_num=10000)
    backward_fraj = track(box, backward_frame_paths, max_new_box_num=10000)
    traj = {'%06d' % fid_int: box}
    for i in range(len(forward_traj)):
        box = forward_traj[i]
        traj['%06d' % (fid_int + i + 1)] = box
    for i in range(len(backward_fraj)):
        box = backward_fraj[i]
        traj['%06d' % (fid_int - i - 1)] = box

    traj_det = {
        'category': det['category'],
        'score': det['score'],
        'trajectory': traj
    }
    return traj_det


if __name__ == '__main__':
    split = 'val'

    traj_det_path = '../evaluation/vidor_%s_object_pred_proc.json' % split
    frame_det_path = 'vidor_%s_object_pred_frame.json'
    data_root = '../data/VidOR/Data/VID/%s' % split

    all_traj_dets = load_trajectory_detections(traj_det_path)
    all_frame_dets = load_frame_detections(frame_det_path)
    sup_frame_dets = supplement_frame_detections(all_traj_dets, all_frame_dets)
    supplement_trajectories(all_traj_dets, sup_frame_dets, data_root)

    output_path = '../evaluation/vidor_%s_object_pred_proc_sup.json' % split
    save_trajectory_detections(output_path, all_traj_dets)

