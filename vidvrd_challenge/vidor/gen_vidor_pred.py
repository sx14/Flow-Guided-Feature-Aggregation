from vidvrd_challenge.evaluation.gen_vidor_pred import gen_vidor_pred

split = 'val'


res_ids = [0]
res_paths = []
for i in res_ids:
    res_path = '..' \
               '/..' \
               '/output' \
               '/fgfa_rfcn' \
               '/vidor_vid' \
               '/resnet_v1_101_flownet_vidor_vid_rfcn_end2end_ohem_%s' \
               '/VID_%s_videos' \
               '/results' \
               '/det_VID_%s_videos%d_all.txt' % (split, split, split, i)
    res_paths.append(res_path)

sav_paths = []
for i in res_ids:
    sav_path = '../evaluation/vidor_%s_object_pred%d.json' % (split, i)
    sav_paths.append(sav_path)

imageset_path = '../../data/VidOR/ImageSets/VID_%s_frames.txt' % split
data_path = '../../data/VidOR/Data/VID/'

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

for i in range(len(res_paths)):
    res_path = res_paths[i]
    sav_path = sav_paths[i]
    gen_vidor_pred(imageset_path, res_path, sav_path, categorys, data_path)

