# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from PIL import ImageFont, ImageDraw, Image

import cv2
import sys
import os
import time
import pickle
import copy
import argparse
import threading

from termcolor import colored
import requests
from collections import deque

import align.detect_face
import align.facenet

import memcache
mc = memcache.Client(['127.0.0.1:12000'], debug=True)

fontpath = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font = ImageFont.truetype(fontpath, 20)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('ip_address', type=str,
                        help='IP address of web camera.')
    parser.add_argument('--face_recognize_threshold', type=float,
                        help='Threshold for face recognize.', default=0.45)
    parser.add_argument('--max_face_number', type=int,
                        help='Number of faces to dectect.', default=1)
    parser.add_argument('--max_frame_rate', type=int,
                        help='Number of FPS threshold.', default=28)
    parser.add_argument('--dangerous_threshold', type=int,
                        help='Threshold of dangerous frame.', default=8)
    parser.add_argument('--cv_test_mode', type=bool,
                        help='Local test with cv windows.', default=False)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=6)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.05)

    return parser.parse_args(argv)


# =================== ARGS ====================
args = parse_arguments(sys.argv[1:])

ip_address = args.ip_address
image_size = args.image_size
frame_margin = args.margin
face_recognize_threshold = args.face_recognize_threshold
max_face_number = args.max_face_number
cv_test_mode = args.cv_test_mode
dangerous_threshold = args.dangerous_threshold
reciprocal_of_max_frame_rate = 1/args.max_frame_rate

minsize = 256
threshold = [0.95, 0.97, 0.99]
factor = 0.709

dangerous_queue = deque(maxlen=dangerous_threshold)

# =================== VIDEO INTERFACE ====================
if cv_test_mode:
    video_capture = cv2.VideoCapture("./media/{}.mp4".format(ip_address))
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 5500)

# =================== CAMERA INTERFACE ====================
else:
    from sdk.xmcext import XMCamera
    cp = XMCamera(ip_address, 34567, "admin", "", "")
    cp.login()
    cp.open()

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))

    with sess.as_default():

        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        print('Loading feature extraction model')
        align.facenet.load_model('model/20180402-114759')

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        with open('model/mlp.pkl', 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print(class_names)

        partial_top = cv2.imread("../extras/images/top.png")
        partial_bottom = cv2.imread("../extras/images/bottom.png")
        partial_safe = cv2.imread("../extras/images/safe.png")

        danger_image = None
        windowNotSet = True
        true_counter = 0

        while True:
            start_time = time.time()

            if cv_test_mode:
                ret, frame = video_capture.read()
            else:
                frame = np.asarray(cp.queryframe('array'))\
                    .reshape(1080, 1920, 3)

            [h, w] = frame.shape[:2]

            # ======================PRE DEAL==========================
            # start_time = time.time()
            # ====================TIMER TIMER=========================

            internal_scaled = []

            real_boxes, _ = align.detect_face.detect_face(
                frame, minsize, pnet, rnet, onet, threshold,
                factor, max_face_number=max_face_number)

            for item in real_boxes:
                item = item.astype(int)
                cropped = frame[item[1]:item[3], item[0]:item[2], :]

                if cropped.shape[0] > 20 and cropped.shape[1] > 20:

                    aligned = cv2.resize(cropped, (image_size, image_size),
                                         interpolation=cv2.INTER_AREA)

                    internal_scaled.append(aligned)

                # cv2.imwrite("./Temp/{}-{}.jpg"
                #             .format(start_time, true_counter), aligned)
                # true_counter += 1

            # ===================internal_scaled======================
            name = []
            confidence = []

            if internal_scaled != []:

                internal_scaled = np.vstack(internal_scaled)\
                    .reshape(-1, image_size, image_size, 3)

                feed_dict = {images_placeholder: internal_scaled,
                             phase_train_placeholder: False}

                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                predictions = model.predict_proba(emb_array)
                predict = np.argmax(predictions, axis=1)

                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)

                for i, _ in enumerate(predictions):
                    # print(predictions)
                    sum_prob = predictions[i, 0] + \
                        predictions[i, 1] + predictions[i, 2]

                    detected_prob = predictions[i, predict[i]]
                    color = (255, 255, 0)

                    item_prob = int(detected_prob*100)
                    face_position = real_boxes[i]

                    res = "非法人员"
                    if detected_prob > face_recognize_threshold:
                        res = class_names[predict[i]]
                        print(class_names[predict[i]], item_prob, sum_prob)

                        # face_position = face_position.astype(int)
                        # cropped = frame[face_position[1]:face_position[3],
                        #                 face_position[0]:face_position[2], :]

                        # cv2.imwrite(
                        #     "./Temp/{}.jpg".format(true_counter), cropped)
                        # true_counter += 1

                    content_string = "{} 可信度-{}%\n{}".format(
                        res, item_prob, time.strftime('%Y-%m-%d %H:%M:%S'))

                    draw.rectangle((face_position[0] - frame_margin,
                                    face_position[1] - frame_margin,
                                    face_position[2] + frame_margin,
                                    face_position[3] + frame_margin), None, color)

                    draw.text((face_position[2] + 15,
                               face_position[1] - frame_margin),
                              content_string, font=font, fill=color)

                    name.append(res)
                    confidence.append(item_prob)

                frame = np.array(img_pil)

            # elapsed_time = time.time() - start_time
            # print('frame time cost: {}'.form通道at(elapsed_time))

            # ===================danger_judge======================

            frame = cv2.resize(frame, (1280, 720),
                               interpolation=cv2.INTER_AREA)

            if '非法人员' in name:
                danger_image = frame
                dangerous_queue.append(1)
            else:
                dangerous_queue.append(0)

            if sum(dangerous_queue) == dangerous_threshold and danger_image is not None:
                partial_medium = cv2.resize(danger_image, (640, 360),
                                            interpolation=cv2.INTER_AREA)

                left = np.concatenate(
                    (partial_top, partial_medium, partial_bottom))

                frame = np.concatenate([left, frame], axis=1)
                mc.set("current_ip", ip_address)
            else:
                frame = np.concatenate([partial_safe, frame], axis=1)

            # start_time = time.time()

            # =====================cv_window========================
            if cv_test_mode:

                if windowNotSet is True:
                    cv2.namedWindow("%s (%d, %d)" %
                                    (ip_address, w, h), cv2.WINDOW_NORMAL)
                    windowNotSet = False

                cv2.imshow("%s (%d, %d)" % (ip_address, w, h), frame)

                k = cv2.waitKey(1) & 0xff
                if k == ord('q') or k == 27:
                    break

            # =====================UPLOADING THREAD========================s
            elif mc.get("current_ip") == ip_address:
                try:
                    requests.post("http://127.0.0.1:6789/upload",
                                  data=cv2.imencode('.jpg', frame)[1].tostring())
                except:
                    print(colored('=>_<= Do not forget to start cds flask server =>_<=',
                                  'yellow'))
                # threading.Thread(
                #     target=lambda: requests.post(
                #         "http://0.0.0.0:6789/upload?address=%s-%s" %
                #         (ip_address, dangerous_flag),
                #         data=cv2.imencode('.jpg', frame)[1].tostring())
                # ).start()

                # c.send("%s-%s-%s" % (ip_address, dangerous_flag,
                #                      cv2.imencode('.jpg', frame)[1].tostring()))

                # requests.post("http://0.0.0.0:6789/upload?address=%s-%d" %
                #               (ip_address, dangerous_flag),
                #               data=cv2.imencode('.jpg', frame)[1].tostring())
                # try:
                #     upload_thread.start()
                # except RuntimeError:
                #     threading.Thread(
                #         target=lambda: requests.post(
                #             "http://0.0.0.0:6789/upload?address=%s" % ip_address,
                #             data=cv2.imencode('.jpg', frame)[1].tostring())
                #     ).start()

            # cv2.imwrite("{}/suspect.jpg"
            #             .format(danger_image_path), frame)

            # true_counter += 1

            # =================TIME MEASURE and ALIGN====================

            res_time = reciprocal_of_max_frame_rate - time.time() + start_time

            if (res_time > 0):
                time.sleep(res_time)

            elapsed_time = time.time() - start_time
            print(colored('%s frame time cost: %s' %
                          (ip_address, elapsed_time), 'blue'))

if cv_test_mode:
    video_capture.release()
    cv2.destroyAllWindows()
else:
    cp.close()
