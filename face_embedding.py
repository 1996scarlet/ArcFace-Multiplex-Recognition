import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import time
import sklearn
from time import sleep

import face_preprocess
import face_image


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
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class EmbeddingModel:
    def __init__(self, args):
        ctx = mx.cpu() if args.gpu == -1 else mx.gpu(args.gpu)
        _vec = args.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))

        self.model = get_model(ctx, image_size, args.model, 'fc1')
        self.face_counter = 0


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


    def get_one_feature(self, nimg):
        return self.get_feature(
            np.transpose(cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB), (2, 0, 1)))
