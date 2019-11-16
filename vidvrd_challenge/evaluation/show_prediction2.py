import os
import json
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt


def good_colors():
    colors = [
        [255, 0, 0],
        [0, 0, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ]
    colors = np.array(colors)
    colors = colors / 255.0
    return colors.tolist()


def random_color():
    color = []
    for i in range(3):
        color.append(random.randint(0, 255) / 255.0)
    return color


def show_trajectories(frame_dir, frame_dets, tid2color, save=False):
    import matplotlib.pyplot as plt

    save_dir = 'temp'
    if save:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)
        else:
            os.mkdir(save_dir)

    fids = sorted([int(fid) for fid in frame_dets if fid != 'viou'])

    plt.figure(0)
    for fid in fids:
        str_fid = '%06d' % fid
        plt.ion()
        plt.axis('off')

        im = plt.imread(os.path.join(frame_dir, str_fid + '.JPEG'))
        plt.imshow(im)

        boxes = frame_dets[str_fid]

        for tid in boxes:
            bbox = boxes[tid]
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=tid2color[tid], linewidth=3.5)
            plt.gca().add_patch(rect)
        plt.show()
        plt.pause(0.0000005)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if save:
            plt.savefig(os.path.join('temp', '%06d.JPEG' % fid), bbox_inches='tight')
        plt.cla()
        plt.cla()
    plt.close()


def show_prediction(video_root, pred_path, vid=None):

    with open(pred_path) as f:
        pred_res = json.load(f)
        vid_res = pred_res['results']

    if vid is not None:
        vid_res = {vid: vid_res[vid]}

    for vid in vid_res:

        frame_dir = os.path.join(video_root, vid)
        frame_list = sorted(os.listdir(frame_dir))
        frame_num = len(frame_list)
        print('>>>> %s [%d] <<<<' % (vid, frame_num))

        video_dets = vid_res[vid]
        video_dets = sorted(video_dets, key=lambda item: item['score'], reverse=True)
        good_dets = {}
        good_tids = []
        for tid, det in enumerate(video_dets):
            if 'viou' not in det['trajectory']:
                continue
            good_tids.append(tid)
            traj = det['trajectory']
            for fid in traj:
                if fid not in good_dets:
                    frame_dets = {}
                    good_dets[fid] = frame_dets
                else:
                    frame_dets = good_dets[fid]

                frame_dets[tid] = traj[fid]

        tid2colors = {}
        colors = good_colors()
        for i, tid in enumerate(good_tids):
            if i < len(colors):
                tid2colors[tid] = colors[i]
            else:
                tid2colors[tid] = random_color()
        show_trajectories(frame_dir, good_dets, tid2colors)






if __name__ == '__main__':
    video_root = '../../data/VidOR/Data/VID/val'
    res_path = 'vidor_val_object_pred_proc_all_2.json'
    vid = u'1025/6163877860'
    show_prediction(video_root, res_path, vid)