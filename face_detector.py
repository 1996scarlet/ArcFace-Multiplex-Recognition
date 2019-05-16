import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import time
from retinaface import RetinaFace
from face_preprocess import preprocess


class DetectorModel:
    def __init__(self, args):
        self.detector = RetinaFace(args.retina_model, 0, args.gpu, 'net3')
        self.threshold = args.threshold
        self.scales = args.scales
        self.max_face_number = args.max_face_number
        self.counter = 0
        self.image_size = args.image_size

    def save_image(self, image):
        cv2.imwrite('./Temp/{}-{}.jpg'.format(time.time(), self.counter),
                    image)
        self.counter += 1

    def get_all_boxes(self, img, save_img=False):
        faces, landmarks = self.detector.detect(img,
                                                self.threshold,
                                                scales=self.scales)

        sorted_index = faces[:, 0].argsort()
        faces = faces[sorted_index]
        landmarks = landmarks[sorted_index]

        aligned = []
        # print('find', faces.shape[0], 'faces')
        for i in range(len(faces[:self.max_face_number])):
            nimg = preprocess(img,
                              faces[i],
                              landmarks[i],
                              image_size=self.image_size)

            if save_img:
                self.save_image(nimg)

            aligned.append(nimg)

        return zip(aligned, faces)