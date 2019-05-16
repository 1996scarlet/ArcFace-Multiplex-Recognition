import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import time
import sklearn
import face_preprocess


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    # print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class EmbeddingModel:
    def __init__(self, args):
        ctx = mx.cpu() if args.gpu == -1 else mx.gpu(args.gpu)
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))

        self.model = get_model(ctx, image_size, args.arcface_model, 'fc1')
        self.face_counter = 0

    def get_one_feature(self, aligned, from_disk=True):
        if from_disk:
            aligned = np.transpose(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB),
                                   (2, 0, 1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding

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
                db = mx.io.DataBatch(data=(data, ))
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
            print('特征转换已完成%2f%%' % (counter * 100 / len(img_paths)))
        return result

    def get_feature_from_raw(self, face_img):
        # face_img is bgr image
        ret = self.detector.detect_face_limited(face_img,
                                                det_type=self.args.det)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T
        # print(bbox)
        # print(points)
        nimg = face_preprocess.preprocess(face_img,
                                          bbox,
                                          points,
                                          image_size='112,112')

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
            db = mx.io.DataBatch(data=(data, ))
            self.model.forward(db, is_train=False)
            _embedding = self.model.get_outputs()[0].asnumpy()
            # print(_embedding.shape)
            if embedding is None:
                embedding = _embedding
            else:
                embedding += _embedding
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding
