import os
import json


def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    i_xmin = max(xmin1, xmin2)
    i_ymin = max(ymin1, ymin2)
    i_xmax = min(xmax1, xmax2)
    i_ymax = min(ymax1, ymax2)
    i_w = i_xmax - i_xmin
    i_h = i_ymax - i_ymin
    iou = 0.0
    if i_w > 0 and i_h > 0:
        u_xmin = min(xmin1, xmin2)
        u_ymin = min(ymin1, ymin2)
        u_xmax = max(xmax1, xmax2)
        u_ymax = max(ymax1, ymax2)
        u_w = u_xmax - u_xmin
        u_h = u_ymax - u_ymin

        iou = (i_w * i_h) * 1.0 / (u_w * u_h)
    return iou


def merge_traj(det1, det2):
    # t1  s1======e1
    # t2      s2======e2
    # merge [s2, e1]
    traj1 = det1['trajectory']
    traj2 = det2['trajectory']
    s1 = int(sorted(traj1.keys())[0])
    e1 = int(sorted(traj1.keys())[-1])
    s2 = int(sorted(traj2.keys())[0])
    e2 = int(sorted(traj2.keys())[-1])
    assert s1 < s2 <= e1 < e2

    cnt = 0
    for i in range(s2, e1+1):
        box1 = traj1['%06d' % i]
        box2 = traj2['%06d' % i]
        iou = cal_iou(box1, box2)

        if iou > 0.5:
            cnt += 1

    merged_det = None
    if cnt > (e1 - s2 + 1) * 0.5:
        # merge
        score1 = det1['score']
        score2 = det2['score']
        if score1 >= score2:
            for i in range(e1+1, e2+1):
                traj1['%6d' % i] = traj2['%6d' % i]
            merged_det = det1
        else:
            for i in range(s1, s2):
                traj2['%6d' % i] = traj1['%6d' % i]
            merged_det = det2

    return merged_det


def connect(video):

    cont = True
    while cont:
        cont = False
        del_det_inds = set()
        new_dets = []
        for i in range(len(video)):
            # assume before
            det1 = video[i]
            traj1 = det1['trajectory']
            cls1 = det1['category']
            s1 = int(sorted(traj1.keys())[0])
            e1 = int(sorted(traj1.keys())[-1])
            for j in range(len(video)):
                # assume after
                if i == j:
                    break

                det2 = video[j]
                traj2 = det2['trajectory']
                cls2 = det2['category']
                s2 = int(sorted(traj2.keys())[0])
                e2 = int(sorted(traj2.keys())[-1])

                if cls1 == cls2 and s1 < s2 <= e1 < e2:
                    # ONLY:
                    #   t1  s1======e1
                    #   t2      s2======e2
                    merged_det = merge_traj(det1, det2)
                    if merged_det is not None:
                        del_det_inds.add(i)
                        del_det_inds.add(j)
                        new_dets.append(merged_det)

        if len(new_dets) > 0:
            cont = True
            for ind in del_det_inds:
                video.pop(ind)
            for det in new_dets:
                video.append(det)






def track(frame_paths, init_box, vis=False):
    # init box: [x1, y1, x2, y2]
    import matplotlib.pyplot as plt
    import cv2

    if vis:
        plt.figure(0)
    new_boxes = []
    tracker = cv2.TrackerKCF_create()
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if i == 0:
            # init tracker
            im_h, im_w, _ = frame.shape
            box = (init_box[0], init_box[1], init_box[2]-init_box[0], init_box[3]-init_box[1])
            status = tracker.init(frame, box)
            box = init_box
            if not status:
                break
        else:
            ok, box = tracker.update(frame)
            box = [int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3])]
            if (not ok) or is_over(box, im_w, im_h):
                break
            new_boxes.append(box)
        if vis:
            plt.ion()
            plt.axis('off')
            print(frame_path)
            frame_show = plt.imread(frame_path)
            plt.imshow(frame_show)

            if box is not None:
                rect = plt.Rectangle((box[0], box[1]),
                                     box[2] - box[0],
                                     box[3] - box[1], fill=False,
                                     edgecolor=[1.0, 0, 0], linewidth=2)
                plt.gca().add_patch(rect)
            plt.show()
            plt.pause(0.01)
            plt.cla()
    if vis:
        plt.close()
    return new_boxes


def is_over(det, im_w, im_h):
    x1, y1, x2, y2 = det
    det_w = x2 - x1
    det_h = y2 - y1

    # check position
    if x1 < 5 or x2 > (im_w - 5):
        # disappear from left/right edge
        if det_h * 1.0 / det_w > 3:
            return True
    if y1 < 5 or y2 > (im_h - 5):
        # disappear from top/bottom edge
        if det_w * 1.0 / det_h > 3:
            return True
    if det_w < 5 and det_h < 5:
        # disappear from center
        return True
    return False


def post_process(res_path, data_root):
    import json
    with open(res_path) as f:
        res = json.load(f)
        all_results = res['results']

    for video_id in all_results:
        video_dir = os.path.join(data_root, video_id)
        frame_list = sorted(os.listdir(video_dir))
        video = all_results[video_id]

        for i, det in enumerate(video):

            w = det['width']
            h = det['height']
            traj = det['trajectory']
            score = det['score']
            boxes = sorted(traj.items(), key=lambda d: d[0])
            det['start'] = boxes[0][0]
            det['end'] = boxes[-1][0]

            start_fid = int(boxes[0][0])
            end_fid = int(boxes[-1][0])

            if start_fid == 0:
                head_is_over = True
            else:
                head_is_over = is_over(boxes[0][1], w, h)

            if end_fid == (len(frame_list)-1):
                tail_is_over = True
            else:
                tail_is_over = is_over(boxes[-1][1], w, h)

            new_boxes = []
            step = 1
            start_frame_id = 0
            if not head_is_over:
                print('[%d] head start: %.2f' % (i, score))
                # tracking backward
                start_frame_id = int(boxes[0][0])
                seg_frames = frame_list[start_frame_id::-1]
                seg_frame_paths = [os.path.join(video_dir, frame_id) for frame_id in seg_frames]
                new_boxes = track(seg_frame_paths, boxes[0][1])
                step = -1

            if not tail_is_over:
                print('[%d] tail start: %.2f' % (i, score))
                # tracking forward
                start_frame_id = int(boxes[-1][0])
                seg_frames = frame_list[start_frame_id:]
                seg_frame_paths = [os.path.join(video_dir, frame_id) for frame_id in seg_frames]
                new_boxes = track(seg_frame_paths, boxes[-1][1])
                step = 1

            if head_is_over and tail_is_over:
                print('[%d] complete traj.' % (i))

            for i in range(len(new_boxes)):
                frame_id = '%06d' % (int(start_frame_id) + step)
                new_box = new_boxes[i]
                traj[frame_id] = new_box

        connect(video)

    res_path1 = res_path[:-5] + '_proc.json'
    with open(res_path1, 'w') as f:
        json.dump(all_results, f)


# test tracker
# with open('../evaluation/imagenet_val_object_pred.json') as f:
#     all_res = json.load(f)
#     all_res = all_res['results']
#
# data_root = '../../data/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000001'
# traj = all_res['ILSVRC2015_val_00000001'][2]['trajectory']
# init_box = traj[sorted(traj.keys())[0]]
# frame_paths = [os.path.join(data_root, fid+'.JPEG') for fid in sorted(traj.keys())]
# track(frame_paths, init_box, traj)