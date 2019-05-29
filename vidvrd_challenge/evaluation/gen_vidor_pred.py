import os
import json

import scipy.io as sio
import numpy as np
import cv2


def gen_vidor_pred(imageset_path, res_paths, save_file_name, category_list, data_root):
    # max_per_video = 50
    score_thr = 0.05

    # load frame-idx
    with open(imageset_path) as f:
        raw_frame_list = f.readlines()
        frame_list = [l.strip() for l in raw_frame_list]
        idx2frame = {}
        for frame_rec in frame_list:
            frame, idx = frame_rec.split(' ')
            idx2frame[int(idx)] = frame

    # load results
    res = []
    for res_path in res_paths:
        with open(res_path) as f:
            lines = f.readlines()
            line_splits = [line.strip().split(' ') for line in lines]
            res_part = [[float(v) for v in line_split]
                        for line_split in line_splits]
            res += res_part

    # output data
    pred_output = {'version': 'VERSION 1.0', 'results': {}}
    pred_results = pred_output['results']  # video_id -> dets

    new_video = True
    for det in res:
        frame_idx, cls_idx, conf, x1, y1, x2, y2, tid = det
        frame_idx = int(frame_idx)
        cls_idx = int(cls_idx)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        tid = int(tid)

        frame_info = idx2frame[frame_idx].split(' ')[0].split('/')
        frame_id = frame_info[-1]
        video_id = '/'.join(frame_info[1:-1])
        frame_path = os.path.join(data_root, idx2frame[frame_idx].split(' ')[0]+'.JPEG')

        if frame_id == '000000':
            if new_video:
                im = cv2.imread(frame_path)
                im_h, im_w, _ = im.shape
                # new video
                video = {'trajectory': {}, 'height': im_h, 'width': im_w}
                pred_results[video_id] = video
                new_video = False
        else:
            new_video = True

        video = pred_results[video_id]
        trajs = video['trajectory']

        if tid in trajs:
            traj = trajs[tid]
        else:
            traj = {}   # fid -> det
            trajs[tid] = traj

        traj[frame_id] = [x1, y1, x2, y2, conf, cls_idx]

    det_num = 0
    for video_id in pred_results:
        video = pred_results[video_id]
        trajs = video['trajectory']
        for tid in trajs:
            traj = trajs[tid]
            det_num += len(traj.keys())
    print(det_num)

    for video_id in pred_results:
        det_num = 0
        video = pred_results[video_id]
        trajs = video['trajectory']

        video_dets = []
        for tid in trajs:
            traj = trajs[tid]
            # print('T[%d]: %d' % (tid, len(traj)))
            det_num += len(traj)
            conf_sum = 0.0  # for avg
            cls_count = np.zeros(len(category_list))   # voting

            for frame_id in traj:
                det = traj[frame_id]
                conf_sum += det[4]
                cls_count[det[5]] += 1
                traj[frame_id] = det[:4]    #[x1,y1,x2,y2]

            cls_ind = np.argmax(cls_count)
            category = category_list[cls_ind]
            score = conf_sum / len(traj.keys())

            if score < score_thr:
                continue

            fids = sorted([int(fid) for fid in traj.keys()])
            video_det = {
                'category': category,
                'score': score,
                'trajectory': traj,
                'org_start_fid': fids[0],
                'org_end_fid': fids[-1],
                'start_fid': fids[0],
                'end_fid': fids[-1],
                'width': video['width'],
                'height': video['height']
            }
            video_dets.append(video_det)

        # video_dets = sorted(video_dets, key=lambda det: det['score'], reverse=True)
        # video_dets = video_dets[:max_per_video]
        print('%s: %d %d' % (video_id, len(video_dets), det_num))
        pred_results[video_id] = video_dets

    curr_dir = os.path.dirname(__file__)
    save_path = os.path.join(curr_dir, save_file_name)
    with open(save_path, 'w') as f:
        json.dump(pred_output, f)




