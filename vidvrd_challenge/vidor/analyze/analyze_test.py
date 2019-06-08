import matplotlib.pyplot as plt

vidor_test_list_path = '../../../data/VidOR/ImageSets/VID_val_videos.txt'


with open(vidor_test_list_path) as f:
    data_list = f.readlines()
    data_list = [line.strip().split(' ') for line in data_list]
video_frame_nums = []
for video_info in data_list:
    video_frame_nums.append(int(video_info[-1]))
print('Test set frame avg: %.2f (%d videos)' % (sum(video_frame_nums) * 1.0 / len(data_list), len(data_list)))

plt.hist(video_frame_nums, histtype='bar', rwidth=0.8)
plt.legend()
plt.xlabel('video_len')
plt.ylabel('videoN')
plt.title('video len')
plt.show()


