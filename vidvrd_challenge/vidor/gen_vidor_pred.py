import json
from vidvrd_challenge.evaluation.gen_vidor_pred import gen_vidor_pred

split = 'val'

# dataset_name = 'VidOR-HOID-mini'
# dataset_name1 = 'vidor_hoid_mini'

dataset_name = 'VidOR'
dataset_name1 = 'vidor_vid'


res_ids = [0]
res_paths = []
for i in res_ids:
    res_path = '..' \
               '/..' \
               '/output' \
               '/fgfa_rfcn' \
               '/%s' \
               '/resnet_v1_101_flownet_%s_rfcn_end2end_ohem_%s' \
               '/VID_%s_videos' \
               '/results' \
               '/det_VID_%s_videos%d_all.txt' % (dataset_name1, dataset_name1, split, split, split, i)
    res_paths.append(res_path)


imageset_path = '../../data/%s/ImageSets/VID_%s_frames.txt' % (dataset_name, split)
data_path = '../../data/%s/Data/VID/' % dataset_name

if dataset_name == 'VidOR':
    categorys = ['__background__',  # always index 0
                 'bread', 'cake', 'dish', 'fruits',
                 'vegetables', 'backpack', 'camera', 'cellphone',
                 'handbag', 'laptop', 'suitcase', 'ball/sports_ball',
                 'bat', 'frisbee', 'racket', 'skateboard',
                 'ski', 'snowboard', 'surfboard', 'toy',
                 'baby_seat', 'bottle', 'chair', 'cup',
                 'electric_fan', 'faucet', 'microwave', 'oven',
                 'refrigerator', 'screen/monitor', 'sink', 'sofa',
                 'stool', 'table', 'toilet', 'guitar',
                 'piano', 'baby_walker', 'bench', 'stop_sign',
                 'traffic_light', 'aircraft', 'bicycle', 'bus/truck',
                 'car', 'motorcycle', 'scooter', 'train',
                 'watercraft', 'crab', 'bird', 'chicken',
                 'duck', 'penguin', 'fish', 'stingray',
                 'crocodile', 'snake', 'turtle', 'antelope',
                 'bear', 'camel', 'cat', 'cattle/cow',
                 'dog', 'elephant', 'hamster/rat', 'horse',
                 'kangaroo', 'leopard', 'lion', 'panda',
                 'pig', 'rabbit', 'sheep/goat', 'squirrel',
                 'tiger', 'adult', 'baby', 'child']

elif dataset_name == 'VidOR-HOID-mini':
    categorys = ["__background__",  # always index 0
                 "adult", "aircraft", "baby", "baby_seat",
                 "baby_walker", "backpack", "ball/sports_ball",
                 "bat", "bench", "bicycle", "bird", "bottle",
                 "cake", "camera", "car", "cat", "cellphone",
                 "chair", "child", "cup", "dish", "dog", "duck",
                 "fruits", "guitar", "handbag", "horse", "laptop",
                 "piano", "rabbit", "racket", "refrigerator",
                 "scooter", "screen/monitor", "skateboard", "ski",
                 "snowboard", "sofa", "stool", "surfboard",
                 "table", "toy", "watercraft"]


sav_paths = []
for i in res_ids:
    sav_path = '../evaluation/%s_%s_object_pred%d.json' % (dataset_name1, split, i)
    sav_paths.append(sav_path)


for i in range(len(res_paths)):
    res_path = res_paths[i]
    sav_path = sav_paths[i]
    gen_vidor_pred(imageset_path, [res_path], sav_path, categorys, data_path)


res_all = None
for sav_path in sav_paths:
    with open(sav_path) as f:
        res = json.load(f)
    if res_all is None:
        res_all = res
    else:
        res_all['results'].update(res['results'])

sav_path = '../evaluation/%s_%s_object_pred_all.json' % (dataset_name1, split)
with open(sav_path, 'w') as f:
    json.dump(res_all, f)

