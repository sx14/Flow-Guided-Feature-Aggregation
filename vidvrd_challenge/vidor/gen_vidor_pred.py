from vidvrd_challenge.utils.gen_vidor_pred import gen_vidor_pred

res_path = '../../output/fgfa_rfcn/vidor_vid/resnet_v1_101_flownet_vidor_vid_rfcn_end2end_ohem/VID_val_videos/val_res.mat'
sav_path = '../evaluation/vidor_val_object_pred.json'
imageset_path = '../../data/VidOR/ImageSets/VID_val_frames.txt'

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


gen_vidor_pred(res_path, sav_path, imageset_path, categorys)

