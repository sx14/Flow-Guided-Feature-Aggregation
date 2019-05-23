import os
from vidvrd_challenge.evaluation.gen_vidor_pred import gen_vidor_pred
from vidvrd_challenge.post_proc.post_proc import post_process

res_path = '..' \
           '/..' \
           '/output' \
           '/fgfa_rfcn' \
           '/imagenet_vid' \
           '/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem' \
           '/VID_val_videos' \
           '/results' \
           '/det_VID_val_videos0_all_sunx.txt'

sav_path = '../evaluation/imagenet_val_object_pred.json'
imageset_path = '../../data/ILSVRC2015/ImageSets/VID_val_videos_eval.txt'
data_root = '../../data/ILSVRC2015/Data/VID/'

categorys = ['__background__',  # always index 0
             'n02691156', 'n02419796', 'n02131653', 'n02834778',
             'n01503061', 'n02924116', 'n02958343', 'n02402425',
             'n02084071', 'n02121808', 'n02503517', 'n02118333',
             'n02510455', 'n02342885', 'n02374451', 'n02129165',
             'n01674464', 'n02484322', 'n03790512', 'n02324045',
             'n02509815', 'n02411705', 'n01726692', 'n02355227',
             'n02129604', 'n04468005', 'n01662784', 'n04530566',
             'n02062744', 'n02391049']


gen_vidor_pred(imageset_path, res_path, sav_path, categorys, data_root)
post_process(sav_path, os.path.join(data_root, 'val'))
