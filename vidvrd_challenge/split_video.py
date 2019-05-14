import os
import sys
import cv2


def split_video_cv2(video_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    video = cv2.VideoCapture(video_path)
    frame_sum = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Frame num: %d' % frame_sum)
    has_next = video.isOpened()
    assert has_next

    # extract and save frames
    fid = 0
    while has_next:
        has_next, frame = video.read()
        frame_path = os.path.join(output_path, '%06d.JPEG' % fid)
        cv2.imwrite(frame_path, frame)
        fid += 1

def split_video_ffmpeg(video_path, output_path):
    cmd = 'ffmpeg -i ' + video_path + ' ' + output_path + '/%06d.JPEG'
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    video_path = '/home/magus/dataset3/VidOR/vidor-dataset/vidor/training/0000/2401075277.mp4'
    output_path = 'frame'
    split_video_cv2(video_path, output_path)