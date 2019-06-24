# coding: utf-8
import math
import cv2
import pickle
import numpy as np
import argparse
import os
import sys
from CImageName import ImageName
import asyncio
import time


def read_pkl_model(mpath):
    with open(mpath, 'rb') as infile:
        (mlp, class_names) = pickle.load(infile)
        print('FR model loadded: ', class_names)
        return mlp, class_names


def start_up_init():
    parser = argparse.ArgumentParser(description='ArcFace Online Test')

    # =================== General ARGS ====================
    parser.add_argument('--max_face_number',
                        type=int,
                        help='同时检测的最大人脸数量',
                        default=16)
    parser.add_argument('--max_frame_rate',
                        type=int,
                        help='Max frame rate',
                        default=25)
    parser.add_argument('--queue_buffer_size',
                        type=int,
                        help='MP Queue size',
                        default=12)
    parser.add_argument('-c', '--usb_camera_code',
                        type=int,
                        nargs='+',
                        help='Code of usb camera. (You can use media file path to test with videos.)',
                        default=[0])
    parser.add_argument('--address_list',
                        type=float,
                        nargs='+',
                        help='IP address of web camera',
                        default=['10.41.0.198', '10.41.0.199'])
    parser.add_argument('--image_size',
                        default='112,112',
                        help='输入特征提取网络的图片大小')
    parser.add_argument('--arcface_model',
                        default='./model/arcface, 0',
                        help='特征提取网络预训练模型路径')
    parser.add_argument('--retina_model',
                        default='./model/R50',
                        help='人脸检测网络预训练模型路径')
    parser.add_argument('--classification',
                        default='./model/mlp.pkl',
                        help='人脸识别分类器模型路径')
    parser.add_argument('--gpu', default=0, type=int, help='GPU设备ID，-1代表使用CPU')
    parser.add_argument('--flip', default=1, type=int, help='是否在训练时进行左右翻转相加操作')
    parser.add_argument('--threshold',
                        default=.6,
                        type=float,
                        help='RetinaNet的人脸检测阈值')
    parser.add_argument('--embedding_threshold',
                        default=.85,
                        type=float,
                        help='需要进行特征提取的人脸可信度阈值')
    parser.add_argument('--scales',
                        type=float,
                        nargs='+',
                        help='RetinaNet的图像缩放系数',
                        default=[1.0])

    return parser.parse_args()


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [
        path for path in os.listdir(path_exp)
        if os.path.isdir(os.path.join(path_exp, path))
    ]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageName(class_name, image_paths))

    return dataset


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for item in dataset:
        image_paths_flat += item.image_paths
        labels_flat += [item.name] * len(item.image_paths)
        # labels_flat.append(item.name)
    return image_paths_flat, labels_flat


def load_data(image_paths):
    nrof_samples = len(image_paths)
    images = [cv2.imread(image_paths[i]) for i in range(nrof_samples)]
    # for i in range(nrof_samples):
    #     img =
    #     images.append(img)
    return images


def encode_image(image, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    return cv2.imencode('.jpg', image, encode_param)[1].tostring()


def draw_points(image, poi, margin=5, color=[255, 255, 0]):
    for index in range(5):
        image[poi[index, 1] - margin:poi[index, 1] + margin, poi[index, 0] -
              margin:poi[index, 0] + margin] = color


def start_up_tools():
    parser = argparse.ArgumentParser(description='Yolo-v3 Online Test')

    # =================== General ARGS ====================
    parser.add_argument('--max_frame_rate',
                        type=int,
                        help='Max frame rate',
                        default=25)
    parser.add_argument('--address_list',
                        type=float,
                        nargs='+',
                        help='IP address of web camera',
                        default=['10.41.0.198', '10.41.0.199'])
    parser.add_argument('--queue_buffer_size',
                        type=int,
                        help='MP Queue size',
                        default=12)
    parser.add_argument('--config',
                        default='./model/tools.cfg',
                        help='Darknet model config')
    parser.add_argument('--weights',
                        default='./model/tools.weights',
                        help='Darknet model weights')
    parser.add_argument('--meta',
                        default='./model/tools.data',
                        help='Darknet model meta')
    parser.add_argument('--threshold',
                        default=.9,
                        type=float,
                        help='Object detection threshold')

    return parser.parse_args()

# Let's compare how fast the implementations are
def time_function(f, *args):
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic
