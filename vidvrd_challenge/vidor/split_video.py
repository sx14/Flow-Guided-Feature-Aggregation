import os
import sys
import cv2


def split_video_cv2(video_path, output_path):
    # provide higher quality frames

    video = cv2.VideoCapture(video_path)
    frame_sum = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Frame num: %d' % frame_sum)
    has_next = video.isOpened()
    assert has_next

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # extract and save frames
    for fid in range(int(frame_sum)):
        has_next, frame = video.read()
        frame_path = os.path.join(output_path, '%06d.JPEG' % fid)
        cv2.imwrite(frame_path, frame)
        fid += 1


def split_video_ffmpeg(video_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cmd = './ffmpeg -i ' + video_path + ' ' + output_path + '/%06d.JPEG -loglevel quiet'
    os.system(cmd)

    # frame id 1 base -> 0 base
    frame_n = len(os.listdir(output_path))
    for fid in range(frame_n):
        org_frame_path = '%s/%06d.JPEG' % (output_path, fid+1)
        new_frame_path = '%s/%06d.JPEG' % (output_path, fid)
        os.renames(org_frame_path, new_frame_path)



if __name__ == '__main__':
    video_path = '/home/magus/dataset3/VidOR/vidor-dataset/vidor/training/1027/2556839256.mp4'
#
    output_path = 'frame'
    split_video_cv2(video_path, output_path)
#
    output_path = 'frame1'
    split_video_ffmpeg(video_path, output_path)