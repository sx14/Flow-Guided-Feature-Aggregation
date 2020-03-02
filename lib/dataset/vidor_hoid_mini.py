"""
VidOR VID database
This class loads ground truth notations from standard VidOR VID XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the ImageNet VID format. Evaluation is based on mAP
criterion.
"""
import copy
import cPickle
import cv2
import os
import numpy as np
import time
from imdb import IMDB
from imagenet_vid_eval import vid_eval
from imagenet_vid_eval_motion import vid_eval_motion
from ds_utils import unique_boxes, filter_small_boxes
from lib.nms.seq_nms import seq_nms, seq_nms_nms, conf_nms
from lib.nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


class VidORHOIDMini(IMDB):
    def __init__(self, image_set, root_path, dataset_path, motion_iou_path, result_path=None,
                 enable_detailed_eval=True):
        """
        fill basic information to initialize imdb
        """
        det_vid = image_set.split('_')[0]
        super(VidORHOIDMini, self).__init__('VidORHOIDMini', image_set, root_path, dataset_path,
                                            result_path)  # set self.name

        self.det_vid = det_vid
        self.root_path = root_path
        self.data_path = dataset_path
        self.motion_iou_path = motion_iou_path
        self.enable_detailed_eval = enable_detailed_eval

        self.classes = ["__background__",  # always index 0
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

        self.classes_map = ["__background__",  # always index 0
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

        self.num_classes = len(self.classes)
        self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
        if len(lines[0]) == 2:
            self.image_set_index = [x[0] for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
        else:
            self.image_set_index = ['%s/%06d' % (x[0], int(x[2])) for x in lines]
            self.pattern = [x[0] + '/%06d' for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
            self.frame_seg_id = [int(x[2]) for x in lines]
            self.frame_seg_len = [int(x[3]) for x in lines]
        # return image_set_index, frame_id

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        if self.det_vid == 'DET':
            image_file = os.path.join(self.data_path, 'Data', 'DET', index + '.JPEG')
        else:
            image_file = os.path.join(self.data_path, 'Data', 'VID', index + '.JPEG')

        # assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self.load_vid_annotation(index) for index in range(0, len(self.image_set_index))]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def load_vid_annotation(self, iindex):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        index = self.image_set_index[iindex]

        import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        roi_rec['frame_id'] = self.frame_id[iindex]
        if hasattr(self, 'frame_seg_id'):
            roi_rec['pattern'] = self.image_path_from_index(self.pattern[iindex])
            roi_rec['frame_seg_id'] = self.frame_seg_id[iindex]
            roi_rec['frame_seg_len'] = self.frame_seg_len[iindex]

        if self.det_vid == 'DET':
            filename = os.path.join(self.data_path, 'Annotations', 'DET', index + '.xml')
        else:
            filename = os.path.join(self.data_path, 'Annotations', 'VID', index + '.xml')

        tree = ET.parse(filename)
        size = tree.find('size')
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width'] = float(size.find('width').text)
        # im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        # assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']

        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        valid_objs = np.zeros((num_objs), dtype=np.bool)

        class_to_index = dict(zip(self.classes_map, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = np.maximum(float(bbox.find('xmin').text), 0)
            y1 = np.maximum(float(bbox.find('ymin').text), 0)
            x2 = np.minimum(float(bbox.find('xmax').text), roi_rec['width'] - 1)
            y2 = np.minimum(float(bbox.find('ymax').text), roi_rec['height'] - 1)
            if not class_to_index.has_key(obj.find('name').text):
                continue
            valid_objs[ix] = True
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        boxes = boxes[valid_objs, :]
        gt_classes = gt_classes[valid_objs]
        overlaps = overlaps[valid_objs, :]

        assert (boxes[:, 2] >= boxes[:, 0]).all()

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

    ###################################################################################################
    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.write_vid_results(detections)
        info = self.do_python_eval()
        return info

    def evaluate_detections_multiprocess_seqnms(self, detections, gpu_id):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results

        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.write_vid_results_multiprocess_seqnms(detections, gpu_id)
        return 1

    def evaluate_detections_multiprocess(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.write_vid_results_multiprocess(detections)
        # info = self.do_python_eval_gen()
        info = self.do_python_eval()
        return info

    def get_result_file_template(self, gpu_id=None):
        """
        :return: a string template
        """
        res_file_folder = os.path.join(self.result_path, 'results')
        if gpu_id is None:
            filename = 'det_' + self.image_set + '_{:s}.txt'
        else:
            filename = 'det_' + self.image_set + str(gpu_id) + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    # def get_result_file_template(self):
    #     """
    #     :return: a string template
    #     """
    #     res_file_folder = os.path.join(self.result_path, 'results')
    #     filename = 'det_' + self.image_set + '_{:s}.txt'
    #     path = os.path.join(res_file_folder, filename)
    #     return path



    def write_vid_results_multiprocess_seqnms(self, detection, gpu_id):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """

        print 'Writing {} VidORVID results file'.format('all')
        filename = self.get_result_file_template(gpu_id).format('all')
        frame_seg_len = copy.deepcopy(self.frame_seg_len)

        # ==== sunx: fix ====
        frame_sum_before = self.frame_id[0] - 1
        if frame_sum_before > 0:
            frame_seg_len.insert(0, frame_sum_before)
        # ===================

        nms = py_nms_wrapper(0.3)
        all_boxes = detection[0]
        frame_ids = detection[1]
        start_idx = 0
        sum_frame_ids = np.cumsum(frame_seg_len)
        first_true_id = frame_ids[0]
        start_video = np.searchsorted(sum_frame_ids, first_true_id)

        # ==== sunx: conf filler ====
        for c in range(len(all_boxes)):
            for f in range(len(all_boxes[c])):
                frame_boxes = all_boxes[c][f]
                if len(frame_boxes) > 0:
                    if c >= 78:
                        frame_boxes = frame_boxes[frame_boxes[:, 4] > 0.2]
                    else:
                        frame_boxes = frame_boxes[frame_boxes[:, 4] > 0.1]
                all_boxes[c][f] = frame_boxes
        # ===========================

        # ==== sunx: multiprocess ====
        print('seq-nms collecting video detections ...')
        videos = []
        for im_ind in range(1, len(frame_ids)):
            true_id = frame_ids[im_ind]
            video_index = np.searchsorted(sum_frame_ids, true_id)
            if video_index != start_video:  # reprensents a new video
                video = [all_boxes[j][start_idx:im_ind] for j in range(1, self.num_classes)]

                # video = conf_nms(video)

                videos.append([video, start_idx, im_ind])

                start_idx = im_ind
                start_video = video_index

        print('seq-nms processing ...')
        t1 = time.time()
        from multiprocessing.pool import Pool as Pool
        from multiprocessing import cpu_count
        cpu_num = min(20, cpu_count())
        print('seq-nms use cpu: %d' % cpu_num)
        pool = Pool(processes=cpu_num)
        results = [pool.apply_async(seq_nms_nms, args=(video[0], 0.7)) for video in videos]
        pool.close()
        pool.join()

        for r, res in enumerate(results):
            videos[r][0] = res.get()

        for video, vid_stt, vid_end in videos:
            for c in xrange(1, self.num_classes):
                all_boxes[c][vid_stt: vid_end] = video[c - 1]

        # the last one
        print('seq-nms last one ...')
        video = [all_boxes[j][start_idx:] for j in range(1, self.num_classes)]
        video = seq_nms_nms(video, 0.7)
        for c in xrange(1, self.num_classes):
            all_boxes[c][start_idx:] = video[c - 1]
        t2 = time.time()
        print('seq-nms finished %.2f s' % (t2 - t1))
        # ==============================

        # for im_ind in range(1, len(frame_ids)):
        #     t = time.time()
        #     true_id = frame_ids[im_ind]
        #     video_index = np.searchsorted(sum_frame_ids, true_id)
        #     if video_index != start_video:  # reprensents a new video
        #         t1 = time.time()
        #         video = [all_boxes[j][start_idx:im_ind] for j in range(1, self.num_classes)]
        #         dets_all = seq_nms(video)
        #         for j in xrange(1, self.num_classes):
        #             for frame_ind, dets in enumerate(dets_all[j - 1]):
        #                 keep = nms(dets)
        #                 all_boxes[j][frame_ind + start_idx] = dets[keep, :]
        #         start_idx = im_ind
        #         start_video = video_index
        #         t2 = time.time()
        #         print 'video_index=', video_index, '  time=', t2 - t1
        #     data_time += time.time() - t
        #     if (im_ind % 100 == 0):
        #         print '{} seq_nms testing {} data {:.4f}s'.format(frame_ids[im_ind - 1], im_ind, data_time / im_ind)
        #
        # # the last video
        # video = [all_boxes[j][start_idx:] for j in range(1, self.num_classes)]
        # dets_all = seq_nms(video)
        # for j in xrange(1, self.num_classes):
        #     for frame_ind, dets in enumerate(dets_all[j - 1]):
        #         keep = nms(dets)
        #         all_boxes[j][frame_ind + start_idx] = dets[keep, :]

        with open(filename, 'wt') as f:
            for im_ind in range(len(frame_ids)):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the imagenet expects 0-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:d} {:d} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f} {:d}\n'.
                                format(frame_ids[im_ind], cls_ind, dets[k, 4],
                                       dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], int(dets[k, -1])))
        return

    def write_vid_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        print 'Writing {} VidORVID results file'.format('all')
        filename = self.get_result_file_template().format('all')
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_set_index):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the imagenet expects 0-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:d} {:d} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.
                                format(self.frame_id[im_ind], cls_ind, dets[k, -1],
                                       dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))

    def write_vid_results_multiprocess(self, detections):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        print 'Writing {} VidORVID results file'.format('all')
        filename = self.get_result_file_template().format('all')
        with open(filename, 'wt') as f:
            for detection in detections:
                all_boxes = detection[0]
                frame_ids = detection[1]
                for im_ind in range(len(frame_ids)):
                    for cls_ind, cls in enumerate(self.classes):
                        if cls == '__background__':
                            continue
                        dets = all_boxes[cls_ind][im_ind]
                        if len(dets) == 0:
                            continue
                        # the imagenet expects 0-based indices
                        for k in range(dets.shape[0]):
                            f.write('{:d} {:d} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.
                                    format(frame_ids[im_ind], cls_ind, dets[k, -1],
                                           dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))

    def do_python_eval(self, gpu_number=None, gpu_id=0):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''
        annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', self.image_set + '_eval.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')

        with open(imageset_file, 'w') as f:
            for i in range(len(self.pattern)):
                for j in range(self.frame_seg_len[i]):
                    f.write((self.pattern[i] % (self.frame_seg_id[i] + j)) + ' ' + str(self.frame_id[i] + j) + '\n')

        if gpu_number != None and gpu_number > 1:
            filenames = []
            for i in range(gpu_number):
                filename = self.get_result_file_template(i).format('all')
                filenames.append(filename)
            multifiles = True  # contains multi cache results of all boxes
        else:
            filenames = self.get_result_file_template(gpu_id).format('all')
            multifiles = False

        ap = vid_eval(multifiles, filenames, annopath, imageset_file, self.classes_map, annocache, ovthresh=0.5)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('AP for {} = {:.4f}'.format(cls, ap[cls_ind - 1]))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap[cls_ind - 1])
        print('Mean AP@0.5 = {:.4f}'.format(np.mean(ap)))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(ap))
        return info_str

    def do_python_eval_gen(self, gpu_number=None):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''
        annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', self.image_set + '_eval.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')

        with open(imageset_file, 'w') as f:
            for i in range(len(self.pattern)):
                for j in range(self.frame_seg_len[i]):
                    f.write((self.pattern[i] % (self.frame_seg_id[i] + j)) + ' ' + str(self.frame_id[i] + j) + '\n')

        if gpu_number != None:
            filenames = []
            for i in range(gpu_number):
                filename = self.get_result_file_template(i).format('all')
                filenames.append(filename)
            multifiles = True  # contains multi cache results of all boxes
        else:
            filenames = self.get_result_file_template().format('all')
            multifiles = False

        if self.enable_detailed_eval:
            # init motion areas and area ranges
            motion_ranges = [[0.0, 1.0], [0.0, 0.7], [0.7, 0.9], [0.9, 1.0]]
            # area_ranges = [[0, 1e5 * 1e5], [0, 50 * 50], [50 * 50, 150 * 150], [150 * 150, 1e5 * 1e5]]
            area_ranges = [[0, 1e5 * 1e5]]
        else:
            motion_ranges = [[0.0, 1.0]]
            area_ranges = [[0, 1e5 * 1e5]]

        ap = vid_eval_motion(multifiles, filenames, annopath, imageset_file, self.classes_map, annocache,
                             self.motion_iou_path,
                             motion_ranges, area_ranges, ovthresh=0.5)

        for motion_index, motion_range in enumerate(motion_ranges):
            for area_index, area_range in enumerate(area_ranges):
                print '================================================='
                print 'motion [{0:.1f} {1:.1f}], area [{2} {3} {4} {5}]'.format(
                    motion_range[0], motion_range[1], np.sqrt(area_range[0]), np.sqrt(area_range[0]),
                    np.sqrt(area_range[1]), np.sqrt(area_range[1]))
                info_str += 'motion [{0:.1f} {1:.1f}], area [{2} {3} {4} {5}]'.format(
                    motion_range[0], motion_range[1], np.sqrt(area_range[0]), np.sqrt(area_range[0]),
                    np.sqrt(area_range[1]), np.sqrt(area_range[1]))
                print('Mean AP@0.5 = {:.4f}'.format(np.mean(
                    [ap[motion_index][area_index][i] for i in range(len(ap[motion_index][area_index])) if
                     ap[motion_index][area_index][i] >= 0])))
                info_str += 'Mean AP@0.5 = {:.4f}'.format(np.mean(
                    [ap[motion_index][area_index][i] for i in range(len(ap[motion_index][area_index])) if
                     ap[motion_index][area_index][i] >= 0]))
        return info_str
