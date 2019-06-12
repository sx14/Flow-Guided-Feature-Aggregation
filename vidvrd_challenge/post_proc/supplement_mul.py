import os
import json
import numpy as np

from post_proc_mul import track

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
    return ious


def get_supplement_frame_detections(all_traj_dets, all_frame_dets):
    for vid in all_frame_dets:
        vid_fram_dets = all_frame_dets[vid]
        vid_traj_dets = all_traj_dets[vid]

        for fid in vid_fram_dets:
            cand_supplement_frame_dets = []
            frame_dets = [det['box'] for det in vid_fram_dets[fid]]
            traj_boxes = [traj_det['trajectory'][fid]
                          for traj_det in vid_traj_dets if fid in traj_det]
            for i, frame_det in enumerate(frame_dets):
                ious = cal_iou(frame_det, traj_boxes)
                ious[ious < 0.7] = 0
                if sum(ious) > 0:
                    cand_supplement_frame_dets.append(frame_dets[i])
            vid_fram_dets[fid] = cand_supplement_frame_dets
    return all_frame_dets


def det2traj(det, frame_num, frame_root):
    fid = det['fid']
    box = det['box']
    fid_int = int(fid)
    forward_frame_paths = [os.path.join(frame_root, '%06d.JPEG' % f) for f in range(fid_int + 1, frame_num)]
    backward_frame_paths = [os.path.join(frame_root, '%06d.JPEG' % f) for f in range(fid_int - 1, -1, -1)]
    forward_traj = track(box, forward_frame_paths)
    backward_fraj = track(box, backward_frame_paths)
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


def track_single_boxes(supplements, data_root):
    results = {}
    for pid_vid in supplements:
        vid_frame_dets = supplements[pid_vid]
        vid_frame_root = os.path.join(data_root, pid_vid)
        frame_num = len(os.listdir(vid_frame_root))

        dets = []
        for fid in vid_frame_dets:
            frame_dets = vid_frame_dets[fid]
            for det in frame_dets:
                det['fid'] = fid
                dets.append(det)

        from multiprocessing.pool import Pool as Pool
        from multiprocessing import cpu_count
        pool = Pool(processes=cpu_count())
        results = [pool.apply_async(det2traj, args=(det, frame_num, vid_frame_root))
                   for d, det in enumerate(dets)]
        pool.close()
        pool.join()

        traj_dets = [results[i].get() for i in range(len(results))]
        results[pid_vid] = traj_dets
    return results



if __name__ == '__main__':
    split = 'val'

    traj_det_path = '../evaluation/vidor_%s_object_pred_proc.json' % split
    frame_det_path = 'vidor_%s_object_pred_frame.json'
    data_root = '../data/VidOR/Data/VID/%s' % split

    all_traj_dets = load_trajectory_detections(traj_det_path)
    all_frame_dets = load_frame_detections(frame_det_path)
    supplement_frame_dets = get_supplement_frame_detections(all_traj_dets, all_frame_dets)
    results = track_single_boxes(supplement_frame_dets, data_root)

    output_path = ''


