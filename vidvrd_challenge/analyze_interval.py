imagenet_train_list_path = '../data/ILSVRC2015/ImageSets/VID_train_15frames.txt'
vidor_train_list_path = '../data/VidOR/ImageSets/VID_train_15frames.txt'

with open(imagenet_train_list_path) as f:
    data_list = f.readlines()

video_frame_ns = []
last = -1
for line in data_list:
    info = line.strip().split(' ')
    video_frame_n = int(info[3])
    if video_frame_n != last:
        video_frame_ns.append(video_frame_n)
        last = video_frame_n

print('ImageNet VID AVG frame num: %.2f' % (sum(video_frame_ns)*1.0/len(video_frame_ns)))
print('ImageNet VID AVG key frame interval: %.2f' % (sum(video_frame_ns)*1.0/len(video_frame_ns)/15))



with open(vidor_train_list_path) as f:
    data_list = f.readlines()

video_frame_ns = []
last = -1
for line in data_list:
    info = line.strip().split(' ')
    video_frame_n = min(int(info[3]), 900)
    if video_frame_n != last:
        video_frame_ns.append(video_frame_n)
        last = video_frame_n

print('VidOR VID AVG frame num: %.2f' % (sum(video_frame_ns)*1.0/len(video_frame_ns)))
print('VidOR VID AVG key frame interval: %.2f' % (sum(video_frame_ns)*1.0/len(video_frame_ns)/15))
