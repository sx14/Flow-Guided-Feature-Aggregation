import os
import time
import json


def show_trajectory(frame_paths, traj, tid):
    import matplotlib.pyplot as plt
    import random
    color = (random.random(), random.random(), random.random())

    plt.figure(tid)
    for i, frame_path in enumerate(frame_paths):
        # print(frame_path.split('/')[-1])
        plt.ion()
        plt.axis('off')

        im = plt.imread(frame_path)
        plt.imshow(im)

        bbox = traj[i]
        if bbox is not None:
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
        plt.show()
        plt.pause(0.001)
        plt.cla()
    plt.close()


def show_prediction(video_root, pred_path, vid=None):

    with open(pred_path) as f:
        pred_res = json.load(f)
        vid_res = pred_res['results']

    if vid is not None:
        vid_res = {vid: vid_res[vid]}

    for vid in vid_res:
        print('>>>> %s <<<<' % vid)
        objs = vid_res[vid]
        for i, obj in enumerate(objs):
            cls = obj['category']
            traj = obj['trajectory']
            score = obj['score']

            frame_dir = os.path.join(video_root, vid)
            frame_list = sorted(os.listdir(frame_dir))
            frame_num = len(frame_list)

            traj_stt_fid = int(sorted(traj.keys())[0])
            traj_end_fid = int(sorted(traj.keys())[-1])
            print('T[%d] %s %.4f [0| %d -> %d |%d]' % (i, cls, score, traj_stt_fid, traj_end_fid, frame_num-1))

            seg_frames = [None for _ in range(len(traj.keys()))]
            for j, fid in enumerate(sorted(traj.keys())):
                seg_frames[j] = fid + '.JPEG'

            seg_frame_paths = [os.path.join(frame_dir, frame_id) for frame_id in seg_frames]
            traj_boxes = [traj[fid] for fid in sorted(traj.keys())]
            show_trajectory(seg_frame_paths, traj_boxes, i)


if __name__ == '__main__':
    video_root = '../../data/VidOR/Data/VID/val'
    res_path = 'vidor_val_object_pred.json'
    vid = u'0004/11566980553'
    show_prediction(video_root, res_path, vid)