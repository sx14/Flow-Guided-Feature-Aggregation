# avg obj num per video
# avg rlt num per video

import os
import json

import numpy as np
import matplotlib.pyplot as plt

spatials = ['above', 'away', 'behind', 'beneath', 'in_front_of', 'inside', 'next_to', 'toward']

anno_root = '../../../data/VidOR/anno/train'
vid_count = 0
rlt_count = 0
obj_count = 0

vid_dur_hist = []

obj_cls_count = {}

obj_num_max = 0
obj_num_min = float('+Inf')
obj_num_sum = 0
obj_stt_fid_hist = []

rlt_num_max = 0
rlt_num_min = float('+Inf')
rlt_num_sum = 0

obj_dur_max = 0
obj_dur_min = float('+Inf')
obj_dur_sum = 0

obj_dur_ratio_sum = 0
obj_dur_ratio_max = 0.0
obj_dur_ratio_min = float('+Inf')


rlt_dur_max = 0
rlt_dur_min = float('+Inf')
rlt_dur_sum = 0
rlt_dur_hist = []

rlt_dur_ratio_sum = 0
rlt_dur_ratio_max = 0.0
rlt_dur_ratio_min = float('+Inf')
rlt_dur_ratio_hist = []

cls_obj_dur_dist = {}


pkg_list = os.listdir(anno_root)
for p, pkg in enumerate(pkg_list):
    print('Count [%d/%d]' % (len(pkg_list), p+1))
    pkg_path = os.path.join(anno_root, pkg)
    for vid_anno_id in os.listdir(pkg_path):
        vid_anno_path = os.path.join(pkg_path, vid_anno_id)
        with open(vid_anno_path) as f:
            vid_anno = json.load(f)
        objs = vid_anno['subject/objects']

        tid2dur = {}
        tid2cls = {}
        tid2stt_fid = {}
        for obj in objs:
            if obj['category'] not in obj_cls_count:
                obj_cls_count[obj['category']] = 1
            else:
                obj_cls_count[obj['category']] += 1

            tid2dur[obj['tid']] = 0
            tid2cls[obj['tid']] = obj['category']
            tid2stt_fid[obj['tid']] = -1

        frame_trajs = vid_anno['trajectories']
        for fid, frame_boxes in enumerate(frame_trajs):
            for box in frame_boxes:
                tid2dur[box['tid']] += 1

                if tid2stt_fid[box['tid']] == -1:
                    tid2stt_fid[box['tid']] = fid

        for tid in tid2dur:
            cls = tid2cls[tid]
            if cls not in cls_obj_dur_dist:
                cls_obj_dur_dist[cls] = []
            cls_obj_dur_dist[cls].append(tid2dur[tid])

            obj_stt_fid_hist.append(tid2stt_fid[tid])

        rlts = vid_anno['relation_instances']
        vid_dur = vid_anno['frame_count']
        vid_dur_hist.append(vid_dur)

        obj_num_max = max(obj_num_max, len(objs))
        obj_num_min = min(obj_num_min, len(objs))

        rlt_num_max = max(rlt_num_max, len(rlts))
        rlt_num_min = min(rlt_num_min, len(rlts))

        obj_num_sum += len(objs)
        rlt_num_sum += len(rlts)
        vid_count += 1

        for rlt in rlts:

            if rlt['predicate'] not in spatials:
                continue

            sid = rlt['subject_tid']
            oid = rlt['object_tid']
            sbj_dur = tid2dur[sid]
            obj_dur = tid2dur[oid]

            stt_fid = rlt['begin_fid']
            end_fid = rlt['end_fid']

            rlt_dur = end_fid - stt_fid + 1
            rlt_dur_min = min(rlt_dur_min, rlt_dur)
            rlt_dur_max = max(rlt_dur_max, rlt_dur)
            rlt_dur_sum += rlt_dur
            rlt_dur_hist.append(rlt_dur)

            # rlt_dur_ratio = rlt_dur * 1.0 / vid_dur
            rlt_dur_ratio = rlt_dur * 1.0 / min(sbj_dur, obj_dur)
            rlt_dur_ratio_min = min(rlt_dur_ratio_min, rlt_dur_ratio)
            rlt_dur_ratio_max = max(rlt_dur_ratio_max, rlt_dur_ratio)
            rlt_dur_ratio_sum += rlt_dur_ratio
            rlt_dur_ratio_hist.append(rlt_dur_ratio)
            rlt_count += 1

        for tid in tid2dur:
            obj_dur = tid2dur[tid]
            obj_dur_min = min(obj_dur_min, obj_dur)
            obj_dur_max = max(obj_dur_max, obj_dur)
            obj_dur_sum += obj_dur

            obj_dur_ratio = obj_dur * 1.0 / vid_dur
            obj_dur_ratio_min = min(obj_dur_ratio_min, obj_dur_ratio)
            obj_dur_ratio_max = max(obj_dur_ratio_max, obj_dur_ratio)
            obj_dur_ratio_sum += obj_dur_ratio
            obj_count += 1

print('-' * 50)

print('obj num: min(%d), max(%d), avg(%.2f)' % (obj_num_min, obj_num_max, obj_num_sum * 1.0 / vid_count))
print('rlt num: min(%d), max(%d), avg(%.2f)' % (rlt_num_min, rlt_num_max, rlt_num_sum * 1.0 / vid_count))

print('-' * 50)

print('rlt dur    : min(%d), max(%d), avg(%.2f)' % (rlt_dur_min, rlt_dur_max, rlt_dur_sum * 1.0 / rlt_count))
print('rlt dur rat: min(%d), max(%d), avg(%.2f)' % (rlt_dur_ratio_min, rlt_dur_ratio_max, rlt_dur_ratio_sum * 1.0 / rlt_count))

print('-' * 50)

print('obj dur    : min(%d), max(%d), avg(%.2f)' % (obj_dur_min, obj_dur_max, obj_dur_sum * 1.0 / obj_count))
print('obj dur rat: min(%d), max(%d), avg(%.2f)' % (obj_dur_ratio_min, obj_dur_ratio_max, obj_dur_ratio_sum * 1.0 / obj_count))

print('-' * 50)

# for cls in cls_obj_dur_dist:
#     print('%s: min(%d) num(%d)' % (cls, min(cls_obj_dur_dist[cls]), len(cls_obj_dur_dist[cls])))
#     print(np.percentile(cls_obj_dur_dist[cls], (25, 50, 75), interpolation='midpoint'))
#     plt.hist(cls_obj_dur_dist[cls], bins=200)
#     plt.show()

obj_num_sum = sum(obj_cls_count.values())
for cls in obj_cls_count:
    cls_count = obj_cls_count[cls]
    print('%s: %.2f' % (cls, cls_count * 1.0 / obj_num_sum))


print('===== video duration distribution =====')
vid_durs = np.array(vid_dur_hist)
dur_percentiles = np.percentile(vid_durs, (25, 50, 75), interpolation='midpoint')
print(dur_percentiles)
plt.hist(vid_dur_hist, 200)
plt.show()


print('===== relation duration distribution =====')
rlt_durs = np.array(rlt_dur_hist)
dur_percentiles = np.percentile(rlt_durs, (25, 50, 75), interpolation='midpoint')
print(dur_percentiles)
plt.hist(rlt_dur_hist, 200)
plt.show()

print('===== relation duration ratio distribution =====')
rlt_dur_ratios = np.array(rlt_dur_ratio_hist)
ration_percentiles = np.percentile(rlt_dur_ratios, (25, 50, 75), interpolation='midpoint')
print(ration_percentiles)
plt.hist(rlt_dur_ratios, 100)
plt.show()

print('===== object start fid distribution =====')
plt.hist(obj_stt_fid_hist, 100)
plt.show()
