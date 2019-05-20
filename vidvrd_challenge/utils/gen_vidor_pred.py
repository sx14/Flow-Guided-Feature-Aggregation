import json

import scipy.io as sio
import numpy as np


def gen_vidor_pred(res_path, save_path, imageset_path, categorys):

    # load frame-idx
    with open(imageset_path) as f:
        raw_frame_list = f.readlines()
        frame_list = [l.strip() for l in raw_frame_list]
        idx2frame = {}
        for frame_rec in frame_list:
            frame, idx = frame_rec.split(' ')
            idx2frame[int(idx)] = frame

    # prediction output data
    pred_output = {'version': 'VERSION 1.0', 'results': {}}
    pred_results = pred_output['results']   # video_id -> dets

    res_list = sio.loadmat(res_path)['res']
    for res in res_list:
        # cls[frame[dets]]
        dets = res[0]
        frame_idxs = res[1]

        for frame_idx in frame_idxs:
            frame_info = idx2frame[frame_idx].split(' ')[0].split('/')
            frame_id = frame_info[-1]
            video_id = frame_info[-2]

            if frame_id == '000000':
                # new video
                trajs = {}  # tid -> traj
                pred_results[video_id] = trajs
            else:
                trajs = pred_results[video_id]

            for cls in range(1, len(dets)):
                frame_dets = dets[cls][frame_idx-1]

                for det in frame_dets:
                    x1, y1, x2, y2, conf, tid = det

                    if tid in trajs:
                        traj = trajs[tid]
                    else:
                        traj = {}   # fid -> det
                        trajs[tid] = traj

                    traj[frame_id] = [x1, y1, x2, y2, conf, cls]

    for video_id in pred_results:
        trajs = pred_results[video_id]
        video_dets = []
        for tid in trajs:
            traj = trajs[tid]

            conf_sum = 0.0  # for avg
            cls_count = np.zeros(len(res_list[0][0]))   # voting

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

        pred_results[video_id] = video_dets

    with open(save_path, 'w') as f:
        json.dump(pred_output, f)




