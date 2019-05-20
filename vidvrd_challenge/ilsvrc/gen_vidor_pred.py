from vidvrd_challenge.utils.gen_vidor_pred import gen_vidor_pred

res_path = '../../output/fgfa_rfcn/imagenet_vid/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem/VID_val_videos/val_res.mat'
sav_path = '../evaluation/imagenet_val_object_pred.json'
imageset_path = '../../data/ILSVRC2015/ImageSets/VID_val_frames.txt'

categorys = ['__background__',  # always index 0
             'n02691156', 'n02419796', 'n02131653', 'n02834778',
             'n01503061', 'n02924116', 'n02958343', 'n02402425',
             'n02084071', 'n02121808', 'n02503517', 'n02118333',
             'n02510455', 'n02342885', 'n02374451', 'n02129165',
             'n01674464', 'n02484322', 'n03790512', 'n02324045',
             'n02509815', 'n02411705', 'n01726692', 'n02355227',
             'n02129604', 'n04468005', 'n01662784', 'n04530566',
             'n02062744', 'n02391049']


gen_vidor_pred(res_path, sav_path, imageset_path, categorys)

