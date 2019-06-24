import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import time
import shutil
from retinaface import RetinaFace


class DetectorModel:
    def __init__(self, args):
        self.detector = RetinaFace(args.retina_model, 0, args.gpu, 'net3')
        self.threshold = args.threshold
        self.scales = args.scales
        self.max_face_number = args.max_face_number
        self.counter = 0
        self.image_size = [112, 112]

    def save_image(self, images):
        for img in images:
            cv2.imwrite(f'./Temp/{time.time()}-{self.counter}.jpg', img)
            self.counter += 1

    def get_all_boxes(self, frame, save_img=False, need_marks=False):
        boxes, landmarks = self.detector.detect(frame,
                                                self.threshold,
                                                scales=self.scales)

        sorted_index = boxes[:, 0].argsort()
        boxes = boxes[sorted_index]
        landmarks = landmarks[sorted_index]

        if need_marks:
            return zip(landmarks, boxes)

        aligned = self.preprocess(frame, boxes, landmarks)

        if save_img:
            self.save_image(aligned)

        return zip(aligned, boxes)

    def get_all_boxes_from_path(self, img_paths, save_img=False):
        for counter, path in enumerate(img_paths):
            base_path, file_name = os.path.split(path)
            if file_name.startswith('cropped'):
                continue

            for face, _ in self.get_all_boxes(cv2.imread(path)):
                cv2.imwrite(f'{base_path}/cropped-{time.time()}.jpg', face)

            shutil.move(path, f'./Temp/raw/{file_name}')
            counter += 1
            print('人脸检测已完成%2f%%' % ((counter * 100) / len(img_paths)))

    def preprocess(self, img, boxes, landmarks, **kwargs):
        aligned = []
        if len(boxes) == len(landmarks):

            for bbox, landmark in zip(boxes, landmarks):
                margin = kwargs.get('margin', 0)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(bbox[0] - margin / 2, 0)
                bb[1] = np.maximum(bbox[1] - margin / 2, 0)
                bb[2] = np.minimum(bbox[2] + margin / 2, img.shape[1])
                bb[3] = np.minimum(bbox[3] + margin / 2, img.shape[0])
                ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
                warped = cv2.resize(ret,
                                    (self.image_size[1], self.image_size[0]))
                aligned.append(warped)

        return aligned