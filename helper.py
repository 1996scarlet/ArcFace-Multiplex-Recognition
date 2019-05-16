# coding: utf-8
import base64
import math
import cv2
import pickle
import numpy as np
import argparse
import os
import sys
from CImageName import ImageName
from scipy.spatial import distance as dist


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
        idxs = np.delete(
            idxs,
            np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

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
    out_data = (out_data - 127.5) * 0.0078125
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
    boundingbox = np.vstack([
        np.round((stride * t_index[1] + 1) / scale),
        np.round((stride * t_index[0] + 1) / scale),
        np.round((stride * t_index[1] + 1 + cellsize) / scale),
        np.round((stride * t_index[0] + 1 + cellsize) / scale), score, reg
    ])

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
    parser.add_argument('--max_face_number',
                        type=int,
                        help='同时检测的最大人脸数量',
                        default=8)
    parser.add_argument('--max_frame_rate', type=int, help='最大FPS', default=25)
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
    parser.add_argument('--det',
                        default=0,
                        type=int,
                        help='设置为1代表使用R+O网络进行检测, 0代表使用P+R+O进行检测')
    parser.add_argument('--flip', default=1, type=int, help='是否在训练时进行左右翻转相加操作')
    parser.add_argument('--threshold',
                        default=0.7,
                        type=float,
                        help='RetinaNet的人脸检测阈值')
    parser.add_argument('--embedding_threshold',
                        default=0.97,
                        type=float,
                        help='需要进行特征提取的人脸可信度阈值')
    # parser.add_argument('-v', '--video_mode', action="store_true",
    #                     help='设置从视频读取帧数据', default=False)
    # parser.add_argument('-c', '--cv_test_mode', action="store_true",
    #                     help='设置本地预览', default=False)
    parser.add_argument('--scales',
                        type=float,
                        nargs='+',
                        help='RetinaNet的图像缩放系数',
                        default=[0.6])

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


def encode_image(image, quality=80):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    return cv2.imencode('.jpg', image, encode_param)[1].tostring()


def draw_points(image, poi, margin=5, color=[255, 255, 0]):
    for index in range(5):
        image[poi[index, 1] - margin:poi[index, 1] + margin, poi[index, 0] -
              margin:poi[index, 0] + margin] = color


K = [
    6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0,
    6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0
]
D = [
    7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0,
    -1.3073460323689292e+000
]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0], [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0], [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0], [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([
        shape[17], shape[21], shape[22], shape[26], shape[36], shape[39],
        shape[42], shape[45], shape[31], shape[35], shape[48], shape[54],
        shape[57], shape[8]
    ])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts,
                                                    cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec,
                                        translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
