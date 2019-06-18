import json
import os

if __name__ == '__main__':

    test_res_path = '../evaluation/vidor_test_object_pred_proc_all.json'
    test_split_dir = '../evaluation/vidor_test_object'

    if not os.path.exists(test_split_dir):
        os.mkdir(test_split_dir)

    with open(test_res_path) as f:
        results = json.load(f)
        results = results['results']

    all_vids = sorted(results.keys())
    split_boundaries = [0, 400, 800, 1200, 1600, 2400]

    for i in range(len(split_boundaries) - 1):
        split_stt_id = split_boundaries[i]
        split_end_id = split_boundaries[i+1]
        split_vids = all_vids[split_stt_id: split_end_id]

        split_resuls = {}
        for vid in split_vids:
            split_resuls[vid] = results[vid]

        split_output = {
            "version": "VERSION 1.0",
            "results": split_resuls
        }

        split_sav_path = os.path.join(test_split_dir, 'vidor_test_object_pred_proc_split_%d.json' % i)
        with open(split_sav_path, 'w') as f:
            json.dump(split_output, f)



