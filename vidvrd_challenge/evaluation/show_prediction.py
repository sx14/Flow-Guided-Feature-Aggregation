import os
import time
import json


def show_trajectory(frame_paths, traj, tid):
    import matplotlib.pyplot as plt
    import random
    color = (random.random(), random.random(), random.random())

    plt.figure(tid)
    for i, frame_path in enumerate(frame_paths):
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
        objs = vid_res[vid]
        for i, obj in enumerate(objs):
            cls = obj['category']
            traj = obj['trajectory']
            score = obj['score']
            print('%s T[%d] %s %.4f' % (vid, i, cls, score))

            frame_dir = os.path.join(video_root, vid)
            frame_list = os.listdir(frame_dir)

            frame_num = len(frame_list)
            traj_show = [None for _ in range(frame_num)]
            for fid in range(frame_num):
                fid_str = '%06d' % fid
                if fid_str in traj:
                    traj_show[fid] = traj[fid_str]

            frame_paths = [os.path.join(frame_dir, frame_id) for frame_id in frame_list]
            show_trajectory(frame_paths, traj_show, i)


if __name__ == '__main__':
    video_root = '../../data/ILSVRC2015/Data/VID/val'
    res_path = 'imagenet_val_object_pred.json'
    vid = u'ILSVRC2015_val_00000001'
    show_prediction(video_root, res_path, vid)