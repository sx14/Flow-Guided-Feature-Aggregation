import os
import shutil
import json

def split_video_ffmpeg(video_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cmd = './ffmpeg -i ' + video_path + ' ' + output_path + '/%06d.JPEG -loglevel quiet'
    os.system(cmd)



if __name__ == '__main__':


    video_path = '/home/magus/dataset3/VidOR/vidor-dataset/vidor/training/0015/6114720259.mp4'
    anno_path = '/home/magus/dataset3/VidOR/vidor-dataset/annotation/training/0015/6114720259.json'
    output_path = 'frame1'

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    split_video_ffmpeg(video_path, output_path)
    anno = json.load(open(anno_path))

    frame_n = len(os.listdir(output_path))
    anno_n = len(anno['trajectories'])
    print frame_n
    print anno_n