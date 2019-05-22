import os
import json

import scipy.io as sio
import numpy as np


def gen_vidor_pred(imageset_path, res_path, save_file_name, categorys):

    # load frame-idx
    with open(imageset_path) as f:
        raw_frame_list = f.readlines()
        frame_list = [l.strip() for l in raw_frame_list]
        idx2frame = {}
        for frame_rec in frame_list:
            frame, idx = frame_rec.split(' ')
            idx2frame[int(idx)] = frame

    # load results
    with open(res_path) as f:
        lines = f.readlines()
        line_splits = [line.strip().split(' ') for line in lines]
        res = [[float(v) for v in line_split]
               for line_split in line_splits]

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
        video_id = frame_info[-2]

        if frame_id == '000000':
            if new_video:
                # new video
                trajs = {}  # tid -> traj
                pred_results[video_id] = trajs
                new_video = False
        else:
            new_video = True

        trajs = pred_results[video_id]

        if tid in trajs:
            traj = trajs[tid]
        else:
            traj = {}   # fid -> det
            trajs[tid] = traj

        traj[frame_id] = [x1, y1, x2, y2, conf, cls_idx]

    det_num = 0
    for vid in pred_results:
        trajs = pred_results[vid]
        for tid in trajs:
            traj = trajs[tid]
            det_num += len(traj.keys())
    print(det_num)

    for video_id in pred_results:
        det_num = 0
        trajs = pred_results[video_id]

        video_dets = []
        for tid in trajs:
            traj = trajs[tid]
            # print('T[%d]: %d' % (tid, len(traj)))
            det_num += len(traj)
            conf_sum = 0.0  # for avg
            cls_count = np.zeros(len(categorys))   # voting

            for frame_id in traj:
                det = traj[frame_id]
                conf_sum += det[4]
                cls_count[det[5]] += 1
                traj[frame_id] = det[:4]    #[x1,y1,x2,y2]

            cls_ind = np.argmax(cls_count)
            category = categorys[cls_ind]
            score = conf_sum / len(traj.keys())

            video_det = {
                'category': category,
                'score': score,
                'trajectory': traj
            }
            video_dets.append(video_det)

        print('%s: %d %d' % (video_id, len(trajs), det_num))
        pred_results[video_id] = video_dets

    curr_dir = os.path.dirname(__file__)
    save_path = os.path.join(curr_dir, save_file_name)
    with open(save_path, 'w') as f:
        json.dump(pred_output, f)




