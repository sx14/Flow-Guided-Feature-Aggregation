import os


def track(frame_paths, init_box, vis=True):
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
            plt.pause(2)
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
    if y1 < 5 or x2 > (im_w - 5):
        # disappear from top/bottom edge
        if det_w * 1.0 / det_h > 3:
            return True
    if det_w < 5 and det_h < 5:
        # disappear from center
        return True
    return False


def connect_trajectory(res_path, data_root):
    import json
    with open(res_path) as f:
        res = json.load(f)
        all_results = res['results']

    for video_id in all_results:
        video_dir = os.path.join(data_root, video_id)
        frame_list = sorted(os.listdir(video_dir))
        video = all_results[video_id]

        for i, det in enumerate(video):

            if i == 2:
                a = 1

            w = det['width']
            h = det['height']
            traj = det['trajectory']
            score = det['score']
            boxes = sorted(traj.items(), key=lambda d: d[0])

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

    res_path1 = res_path[:-5] + '_track.json'
    with open(res_path1, 'w') as f:
        json.dump(all_results, f)

