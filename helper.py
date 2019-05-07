# coding: utf-8
import math
import cv2
import pickle
import numpy as np
import argparse
import os
import sys
from CImageName import ImageName


def nms(boxes, overlap_threshold, mode='Union'):
    """
        non max suppression

    Parameters:
    ----------
        box: numpy array n x 5
            input bbox array
        overlap_threshold: float number
            threshold of overlap
        mode: float number
            how to compute overlap ratio, 'Union' or 'Min'
    Returns:
    -------
        index array of the selected bbox
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        if mode == 'Min':
            overlap = inter / np.minimum(area[i], area[idxs[:last]])
        else:
            overlap = inter / (area[i] + area[idxs[:last]] - inter)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))

    return pick


def adjust_input(in_data):
    """
        adjust the input from (h, w, c) to ( 1, c, h, w) for network input

    Parameters:
    ----------
        in_data: numpy array of shape (h, w, c)
            input data
    Returns:
    -------
        out_data: numpy array of shape (1, c, h, w)
            reshaped array
    """
    if in_data.dtype is not np.dtype('float32'):
        out_data = in_data.astype(np.float32)
    else:
        out_data = in_data

    out_data = out_data.transpose((2, 0, 1))
    out_data = np.expand_dims(out_data, 0)
    out_data = (out_data - 127.5)*0.0078125
    return out_data


def generate_bbox(map, reg, scale, threshold):
    """
        generate bbox from feature map
    Parameters:
    ----------
        map: numpy array , n x m x 1
            detect score for each position
        reg: numpy array , n x m x 4
            bbox
        scale: float number
            scale of this detection
        threshold: float number
            detect threshold
    Returns:
    -------
        bbox array
    """
    stride = 2
    cellsize = 12

    t_index = np.where(map > threshold)

    # find nothing
    if t_index[0].size == 0:
        return np.array([])

    dx1, dy1, dx2, dy2 = [reg[0, i, t_index[0], t_index[1]] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = map[t_index[0], t_index[1]]
    boundingbox = np.vstack([np.round((stride*t_index[1]+1)/scale),
                             np.round((stride*t_index[0]+1)/scale),
                             np.round((stride*t_index[1]+1+cellsize)/scale),
                             np.round((stride*t_index[0]+1+cellsize)/scale),
                             score,
                             reg])

    return boundingbox.T


def detect_first_stage(img, net, scale, threshold):
    """
        run PNet for first stage

    Parameters:
    ----------
        img: numpy array, bgr order
            input image
        scale: float number
            how much should the input image scale
        net: PNet
            worker
    Returns:
    -------
        total_boxes : bboxes
    """
    height, width, _ = img.shape
    hs = int(math.ceil(height * scale))
    ws = int(math.ceil(width * scale))

    im_data = cv2.resize(img, (ws, hs))

    # adjust for the network input
    input_buf = adjust_input(im_data)
    output = net.predict(input_buf)
    boxes = generate_bbox(output[1][0, 1, :, :], output[0], scale, threshold)

    if boxes.size == 0:
        return None

    # nms
    pick = nms(boxes[:, 0:5], 0.5, mode='Union')
    boxes = boxes[pick]
    return boxes


def detect_first_stage_warpper(args):
    return detect_first_stage(*args)


def read_pkl_model(mpath):
    with open(mpath, 'rb') as infile:
        (mlp, class_names) = pickle.load(infile)
        print('FR model loadded: ', class_names)
        return mlp, class_names


def start_up_init():
    parser = argparse.ArgumentParser(description='ArcFace Online Test')

    # =================== General ARGS ====================
    # if not train_mode:
    #     parser.add_argument('ip_address', type=str,
    #                         help='相机的IP地址或测试用视频文件名')
    # parser.add_argument('--face_recognize_threshold', type=float,
    #                     help='可疑人员识别阈值', default=0.95)
    parser.add_argument('--max_face_number', type=int,
                        help='同时检测的最大人脸数量', default=8)
    parser.add_argument('--max_frame_rate', type=int,
                        help='最大FPS', default=25)
    parser.add_argument('--image-size', default='112,112',
                        help='输入特征提取网络的图片大小')
    # parser.add_argument('--dangerous_threshold', type=int,
    #                     help='1/2报警窗口长度', default=16)
    parser.add_argument('--model', default='./model-r100-ii/arcface,0',
                        help='特征提取网络预训练模型路径')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU设备ID，-1代表使用CPU')
    parser.add_argument('--det', default=0, type=int,
                        help='设置为1代表使用R+O网络进行检测, 0代表使用P+R+O进行检测')
    parser.add_argument('--flip', default=1, type=int,
                        help='是否在训练时进行左右翻转相加操作')
    parser.add_argument('--threshold', default=1.24, type=float,
                        help='空间向量距离阈值')
    # parser.add_argument('-v', '--video_mode', action="store_true",
    #                     help='设置从视频读取帧数据', default=False)
    # parser.add_argument('-c', '--cv_test_mode', action="store_true",
    #                     help='设置本地预览', default=False)
    parser.add_argument('--mtcnn_minsize', type=int,
                        help='mtcnn最小检测框的尺寸（越小检测精度越高）', default=48)
    parser.add_argument('--mtcnn_factor', type=float,
                        help='mtcnn图像缩放系数（关联图像金字塔层数，越大检测精度越高）', default=0.809)
    parser.add_argument('--mtcnn_threshold', type=float, nargs='+',
                        help='mtcnn三层阈值', default=[0.6, 0.7, 0.92])

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
    classes = [path for path in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, path))]
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


def encode_image(image, quality=80):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    return cv2.imencode('.jpg', image, encode_param)[1].tostring()
