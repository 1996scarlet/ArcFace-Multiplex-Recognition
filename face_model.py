from scipy import misc
import sys
import os
import argparse
import numpy as np
import mxnet as mx
import random
import cv2
import time
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector

import face_preprocess
import face_image


def do_flip(data):
    for idx in xrange(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, args):
        self.args = args
        if args.gpu == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(args.gpu)
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.model = None
        if len(args.model) > 0:
            self.model = get_model(ctx, image_size, args.model, 'fc1')

        self.threshold = args.threshold
        self.det_minsize = args.mtcnn_minsize
        self.det_threshold = args.mtcnn_threshold
        self.det_factor = args.mtcnn_factor
        self.image_size = image_size
        mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
        if args.det == 0:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1,
                                     minsize=self.det_minsize, factor=self.det_factor,
                                     accurate_landmark=True, threshold=self.det_threshold)
        else:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx,
                                     num_worker=1, accurate_landmark=True, threshold=[0.0, 0.0, 0.2])
        self.detector = detector
        self.face_counter = 0

    def get_input(self, face_img):
        ret = self.detector.detect_face(face_img, det_type=self.args.det)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T
        # print(bbox)
        # print(points)
        nimg = face_preprocess.preprocess(
            face_img, bbox, points, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def get_all_input(self, face_img, save_img=False, face_num=10):
        ret = self.detector.detect_face(face_img, det_type=self.args.det)
        if ret is None:
            return []

        bbox, points = ret

        sorted_index = bbox[:, 0].argsort()
        bbox = bbox[sorted_index]
        points = points[sorted_index]

        # print(bbox)
        # print(points)

        if bbox.shape[0] == 0:
            return None

        aligned = []
        for index in range(0, len(bbox[:face_num])):
            item_bbox = bbox[index, 0:4]
            item_points = points[index, :].reshape((2, 5)).T
            # print(bbox)
            # print(points)
            nimg = face_preprocess.preprocess(
                face_img, item_bbox, item_points, image_size='112,112')

            if save_img:
                cv2.imwrite('./Temp/{}-{}.jpg'.format(time.time(),
                                                      self.face_counter), nimg)
                # print(self.face_counter)
                self.face_counter += 1

            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            aligned.append(np.transpose(nimg, (2, 0, 1)))

        # print(aligned)
        return zip(aligned, bbox)

    def get_feature(self, aligned, from_disk=False):
        if from_disk:
            aligned = np.transpose(cv2.cvtColor(
                aligned, cv2.COLOR_BGR2RGB), (2, 0, 1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding

    def get_all_features(self, face_img, save_img=False, face_num=10):
        ret = self.detector.detect_face(face_img, det_type=self.args.det)
        if ret is None:
            return []

        bbox, points = ret

        sorted_index = bbox[:, 0].argsort()
        bbox = bbox[sorted_index]
        points = points[sorted_index]

        # print(bbox)
        # print(points)

        if bbox.shape[0] == 0:
            return None

        aligned = []
        features = []
        for index in range(0, len(bbox[:face_num])):
            item_bbox = bbox[index, 0:4]
            item_points = points[index, :].reshape((2, 5)).T
            # print(bbox)
            # print(points)
            nimg = face_preprocess.preprocess(
                face_img, item_bbox, item_points, image_size='112,112')

            if save_img:
                cv2.imwrite('./Temp/{}-{}.jpg'.format(time.time(),
                                                      self.face_counter), nimg)
                # print(self.face_counter)
                self.face_counter += 1
            aligned.append(nimg)
            features.append(self.get_feature(
                np.transpose(
                    cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB), (2, 0, 1))))

        # print(aligned)
        return zip(aligned, features, bbox)
