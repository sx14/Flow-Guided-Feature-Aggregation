# ====================================================
# @Time    : 2/25/19 9:11 AM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : video_object_detection.py
# ====================================================
import numpy as np

from vidvrd_challenge.evaluation.common import voc_ap, iou


def trajectory_overlap(gt_trajs, pred_traj):
    """
    Calculate overlap among trajectories
    :param gt_trajs:
    :param pred_traj:
    :param thresh_s:
    :return:
    """
    max_overlaps = [0] * len(gt_trajs)
    thresh = 0.5
    for t, gt_traj in enumerate(gt_trajs):
        top1 = 0
        total = len(set(gt_traj.keys()) | set(pred_traj.keys()))
        gt_len = len(gt_traj.keys())
        for i, fid in enumerate(gt_traj):
            if fid not in pred_traj:
                continue
            sIoU = iou(gt_traj[fid], pred_traj[fid])
            if sIoU >= thresh:
                top1 += 1

        # tIoU = (top1 + top2 + top3) * 1.0 / (3 * total)
        tIoU = (top1) * 1.0 / (gt_len)

        if tIoU > max_overlaps[t]:
            max_overlaps[t] = tIoU

    return max_overlaps


def evaluate(gt, pred, use_07_metric=True, thresh_t=0.8):
    MAX_PER_VIDEO = 20
    """
    Evaluate the predictions
    """
    gt_classes = set()
    for vid, tracks in gt.items():
        vid_classes = set()
        for traj in tracks:
            gt_classes.add(traj['category'])
            vid_classes.add(traj['category'])
        # print(vid)
        # print(list(vid_classes))
    gt_class_num = len(gt_classes)

    result_class = dict()
    for vid, tracks in pred.items():
        tracks = sorted(tracks, key=lambda traj: traj['score'], reverse=True)
        tracks = tracks[:MAX_PER_VIDEO]
        for traj in tracks:
            if traj['category'] not in result_class:
                result_class[traj['category']] = [[vid, traj['score'], traj['trajectory']]]
            else:
                result_class[traj['category']].append([vid, traj['score'], traj['trajectory']])

    rec_class = dict()
    print('Computing average recall AR over {} classes...'.format(gt_class_num))
    for c in gt_classes:

        npos = 0
        class_recs = {}

        for vid in gt:
            # print(vid)
            gt_trajs = [trk['trajectory'] for trk in gt[vid] if trk['category'] == c]
            det = [False] * len(gt_trajs)
            npos += len(gt_trajs)
            class_recs[vid] = {'trajectories': gt_trajs, 'det': det}

        if c not in result_class:
            rec_class[c] = [0, npos]
            continue

        trajs = result_class[c]
        vids = [trj[0] for trj in trajs]
        scores = np.array([trj[1] for trj in trajs])
        trajectories = [trj[2] for trj in trajs]

        nd = len(vids)
        sorted_inds = np.argsort(-scores)
        sorted_vids = [vids[id] for id in sorted_inds]
        sorted_scrs = [scores[id] for id in sorted_inds]
        sorted_traj = [trajectories[id] for id in sorted_inds]

        hit_pred_scr_dist = []
        hit_pred_len_dist = []
        gt_len_dist = []

        for d in range(nd):
            R = class_recs[sorted_vids[d]]
            gt_trajs = R['trajectories']
            pred_traj = sorted_traj[d]
            max_overlaps = trajectory_overlap(gt_trajs, pred_traj)

            for g, max_overlap in enumerate(max_overlaps):
                if max_overlap >= thresh_t:
                    pred_traj['viou'] = max_overlap
                    R['det'][g] = True
                    hit_pred_scr_dist.append(sorted_scrs[d])
                    hit_pred_len_dist.append(len(sorted_traj[d].keys()))

        gt_sum = 0
        gt_hit = 0
        for vid in class_recs:
            gt_sum += len(class_recs[vid]['det'])
            for hit in class_recs[vid]['det']:
                if hit:
                    gt_hit += 1
            for traj in class_recs[vid]['trajectories']:
                gt_len_dist.append(len(traj.keys()))
        rec_class[c] = [gt_hit, gt_sum]

        print('=' * 50)
        print('%s: rec(%.2f) gt(%d)' % (c, rec_class[c][0] * 1.0 / rec_class[c][1], rec_class[c][1]))

        if len(hit_pred_scr_dist) > 0:
            len_percentiles = np.percentile(hit_pred_len_dist, (10, 25, 50, 75), interpolation='midpoint')
            print('%s: min_len(%d) avg_len(%.2f)' % (c,
                                                     min(hit_pred_len_dist),
                                                     sum(hit_pred_len_dist) * 1.0 / max(len(hit_pred_len_dist), 1)))
            print(len_percentiles)
            print('%s: min_scr(%.2f) avg_scr(%.2f)' % (c,
                                                       min(hit_pred_scr_dist),
                                                       sum(hit_pred_scr_dist) * 1.0 / max(len(hit_pred_scr_dist), 1)))
            scr_percentiles = np.percentile(hit_pred_scr_dist, (10, 25, 50, 75), interpolation='midpoint')
            print(scr_percentiles)

    # compute mean recall and print
    output = []
    print('=' * 36)
    output.append('=' * 36 + '\n')
    rec_class = sorted(rec_class.items(), key=lambda rec_cls: rec_cls[0])
    total_hit = 0
    total_gt = 0
    for i, (category, rec) in enumerate(rec_class):
        print('{:>2}{:>20}\t{:.4f}\t{:>4}'.format(i + 1, category, rec[0] * 1.0 / rec[1], rec[1]))
        output.append('{:>2}{:>20}\t{:.4f}\t{:>4}\n'.format(i + 1, category, rec[0] * 1.0 / rec[1], rec[1]))
        total_hit += rec[0]
        total_gt += rec[1]
    mean_rec = total_hit * 1.0 / total_gt
    print('=' * 36)
    output.append('=' * 36 + '\n')
    print('{:>22}\t{:.4f}\t{:>4}\n'.format('mean Recall', mean_rec, total_gt))
    output.append('{:>22}\t{:.4f}\t{:>4}\n'.format('mean Recall', mean_rec, total_gt))

    return mean_rec, rec_class, output


if __name__ == "__main__":
    """
    You can directly run this script from the parent directory, e.g.,
    python -m evaluation.video_object_detection val_object_groundtruth.json val_object_prediction.json
    """
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Video object detection evaluation.')
    parser.add_argument('groundtruth', type=str, help='A ground truth JSON file generated by yourself')
    parser.add_argument('prediction', type=str, help='A prediction file')
    args = parser.parse_args()

    print('Loading ground truth from {}'.format(args.groundtruth))
    with open(args.groundtruth, 'r') as fp:
        gt = json.load(fp)
    print('Number of videos in ground truth: {}'.format(len(gt)))

    print('Loading prediction from {}'.format(args.prediction))
    with open(args.prediction, 'r') as fp:
        pred = json.load(fp)
    print('Number of videos in prediction: {}'.format(len(pred['results'])))

    mean_rec, rec_class, output = evaluate(gt, pred['results'])
    with open('vidor_val_object_segment_results.txt', 'w') as f:
        f.writelines(output)

    with open(args.prediction, 'w') as fp:
        print('updating predictions...')
        json.dump(pred, fp)