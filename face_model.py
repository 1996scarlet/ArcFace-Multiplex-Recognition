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
    for idx in range(data.shape[0]):
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
        ctx = mx.cpu() if args.gpu == -1 else mx.gpu(args.gpu)
        #if args.gpu == -1:
        #    ctx = mx.cpu()
        #else:
        #    ctx = mx.gpu(args.gpu)
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))

        self.max_face_number = args.max_face_number
        self.threshold = args.threshold
        self.image_size = image_size

        if args.only_detector:
            self.det_minsize = args.mtcnn_minsize
            self.det_threshold = args.mtcnn_threshold
            self.det_factor = args.mtcnn_factor
            mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
            if args.det == 0:
                detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1,
                                        minsize=self.det_minsize, factor=self.det_factor,
                                        accurate_landmark=True, threshold=self.det_threshold)
            else:
                detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx,
                                        num_worker=1, accurate_landmark=True, threshold=[0.0, 0.0, 0.2])
            self.detector = detector
        else:
            self.model = None
            if len(args.model) > 0:
                self.model = get_model(ctx, image_size, args.model, 'fc1')

        self.face_counter = 0

    # def __init__(self, args):
    #     self.args = args
    #     if args.gpu == -1:
    #         ctx = mx.cpu()
    #     else:
    #         ctx = mx.gpu(args.gpu)
    #     _vec = args.image_size.split(',')
    #     assert len(_vec) == 2
    #     image_size = (int(_vec[0]), int(_vec[1]))
    #     self.model = None
    #     if len(args.model) > 0:
    #         self.model = get_model(ctx, image_size, args.model, 'fc1')

    #     self.max_face_number = args.max_face_number
    #     self.threshold = args.threshold
    #     self.det_minsize = args.mtcnn_minsize
    #     self.det_threshold = args.mtcnn_threshold
    #     self.det_factor = args.mtcnn_factor
    #     self.image_size = image_size
    #     mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    #     if args.det == 0:
    #         detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1,
    #                                  minsize=self.det_minsize, factor=self.det_factor,
    #                                  accurate_landmark=True, threshold=self.det_threshold)
    #     else:
    #         detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx,
    #                                  num_worker=1, accurate_landmark=True, threshold=[0.0, 0.0, 0.2])
    #     self.detector = detector
    #     self.face_counter = 0

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

    def get_all_input(self, face_img, save_img=False):
        face_num = self.max_face_number
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

    def get_all_boxes(self, face_img, save_img=False):
        face_num = self.max_face_number
        ret = self.detector.detect_face(face_img, det_type=self.args.det)
        if ret is None:
            return []

        bbox, points = ret

        sorted_index = bbox[:, 0].argsort()
        bbox = bbox[sorted_index]
        points = points[sorted_index]

        aligned = []
        for index in range(0, len(bbox[:face_num])):
            item_bbox = bbox[index, 0:4]
            item_points = points[index, :].reshape((2, 5)).T
            nimg = face_preprocess.preprocess(
                face_img, item_bbox, item_points, image_size='112,112')

            if save_img:
                cv2.imwrite('./Temp/{}-{}.jpg'.
                            format(time.time(), self.face_counter), nimg)
                self.face_counter += 1

            aligned.append(nimg)

        return zip(aligned, bbox)

    def get_all_features(self, face_img, save_img=False):
        face_num = self.max_face_number
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

    def get_one_feature(self, nimg):
        return self.get_feature(
            np.transpose(cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB), (2, 0, 1)))

    def get_features_from_path(self, img_paths):
        result = []
        for counter, path in enumerate(img_paths):
            # print(type(len(img_paths)))
            # print(type(counter))
            nimg = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            aligned = np.transpose(nimg, (2, 0, 1))
            embedding = None
            for flipid in [0, 1]:
                if flipid == 1 and self.args.flip == 1:
                    do_flip(aligned)

                input_blob = np.expand_dims(aligned, axis=0)
                data = mx.nd.array(input_blob)
                db = mx.io.DataBatch(data=(data,))
                self.model.forward(db, is_train=False)
                _embedding = self.model.get_outputs()[0].asnumpy()
                # print(_embedding.shape)
                if embedding is None:
                    embedding = _embedding
                else:
                    embedding += _embedding
            embedding = sklearn.preprocessing.normalize(embedding).flatten()
            result.append(embedding)
            # print()
            print('特征转换已完成%2f%%' % (counter*100/len(img_paths)))
        return result

    def get_feature_from_raw(self, face_img):
        # face_img is bgr image
        ret = self.detector.detect_face_limited(
            face_img, det_type=self.args.det)
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

        # cv2.imshow(' ', nimg)
        # cv2.waitKey(0)

        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        # print(nimg.shape)
        embedding = None
        for flipid in [0, 1]:
            if flipid == 1:
                if self.args.flip == 0:
                    break
                do_flip(aligned)
            input_blob = np.expand_dims(aligned, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            self.model.forward(db, is_train=False)
            _embedding = self.model.get_outputs()[0].asnumpy()
            # print(_embedding.shape)
            if embedding is None:
                embedding = _embedding
            else:
                embedding += _embedding
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding

# frame = cv2.imread('./t1.jpg')
# aligned = arcface.get_all_input(frame)

# for face in aligned:
#         f1 = arcface.get_feature(face)
#         print(f1[:10])

# [h, w] = frame.shape[:2]
# "%s (%d, %d)" % (ip_address, w, h)
