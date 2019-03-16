import face_model
import argparse
import cv2
import sys
import numpy as np
import time
from termcolor import colored
import pickle
from collections import deque
from dataclasses import dataclass

# =================== Data Class For Left Panel ====================
@dataclass
class HeadPicture:
    image: object
    prob: str
    time: str
    name: str = 'Suspect Person'

    def draw_text(self, frame, x_offset, y_offset, color):
        frame[y_offset:y_offset + self.image.shape[0],
              x_offset - 128:x_offset - 128+self.image.shape[1]] = self.image
        cv2.putText(frame, self.time, (x_offset, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(frame, self.name, (x_offset, y_offset + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, self.prob, (x_offset, y_offset + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('ip_address', type=str,
                    help='IP address of web camera.')
parser.add_argument('--face_recognize_threshold', type=float,
                    help='Threshold for face recognize.', default=0.95)
parser.add_argument('--max_face_number', type=int,
                    help='Number of faces to dectect.', default=8)
parser.add_argument('--max_frame_rate', type=int,
                    help='Number of FPS threshold.', default=28)
parser.add_argument('--image-size', default='112,112',
                    help='')
parser.add_argument('--dangerous_threshold', type=int,
                    help='Threshold of dangerous frame.', default=8)
parser.add_argument('--model', default='./model-r100-ii/model,0',
                    help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int,
                    help='gpu id, -1 for cpu')
parser.add_argument('--det', default=0, type=int,
                    help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int,
                    help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float,
                    help='ver dist threshold')
parser.add_argument('--cv_test_mode', type=bool,
                    help='Local test with cv windows.', default=False)
parser.add_argument('--mtcnn_minsize', type=int,
                    help='mtcnn minsize.', default=50)
parser.add_argument('--mtcnn_factor', type=float,
                    help='mtcnn minsize.', default=0.709)
parser.add_argument('--mtcnn_threshold', type=list,
                    help='mtcnn minsize.', default=[0.6, 0.7, 0.9])

# =================== ARGS ====================
args = parser.parse_args()
arcface = face_model.FaceModel(args)

face_recognize_threshold = args.face_recognize_threshold
cv_test_mode = args.cv_test_mode
ip_address = args.ip_address
dangerous_threshold = args.dangerous_threshold
max_face_number = args.max_face_number
reciprocal_of_max_frame_rate = 1/args.max_frame_rate

windowNotSet = True
partial_safe = cv2.imread("./Pixel/LBG.png")
# dangerous_queue = deque(maxlen=dangerous_threshold)
suspects_queue = deque(maxlen=4)
workers_queue = deque(maxlen=4)

# =================== VIDEO INTERFACE ====================
if cv_test_mode:
    video_capture = cv2.VideoCapture(
        "../Single_Face/static/single_core/media/{}".format(ip_address))
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 300)

# =================== CAMERA INTERFACE ====================
else:
    frame = cv2.imread('./t1.jpg')
    aligned = arcface.get_all_input(frame)

    for face in aligned:
        f1 = arcface.get_feature(face)
        print(f1[:10])

# =================== FR MODEL ====================
with open('./model-mlp/mlp.pkl', 'rb') as infile:
    (mlp, class_names) = pickle.load(infile)
    print(class_names)

# =================== ETERNAL LOOP ====================
while True:

    if cv_test_mode:
        ret, frame = video_capture.read()
    else:
        frame = []

    # [h, w] = frame.shape[:2]
    # "%s (%d, %d)" % (ip_address, w, h)
    # ======================PRE DEAL==========================
    # start_time = time.time()
    # ====================TIMER TIMER=========================

    start_time = time.time()
    persons = arcface.get_all_features(
        frame, save_img=False, face_num=max_face_number)
    mid_time = time.time()
    print(colored('face detection and features embedding cost -> %.2f ' %
                  (mid_time - start_time), 'yellow'))

    dt = time.strftime('%Y-%m-%d %H:%M:%S')
    for img, vec, box in persons:
        box = box.astype(int)
        predict = mlp.predict_proba([vec])
        prob = predict.max(1)

        if prob < face_recognize_threshold:
            colors = [0, 0, 255]
            suspects_queue.append(HeadPicture(
                image=img,
                time=dt,
                prob='Suspicious: %.2f' % (100-prob*100)))

        else:
            colors = [0, 255, 0]
            workers_queue.append(HeadPicture(
                image=img,
                time=dt,
                name=class_names[predict.argmax(1)[0]],
                prob='Credibility: %.2f' % (prob * 100)))

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colors, 2)

    # =====================FRAME Process========================
    frame = np.concatenate([partial_safe, cv2.resize(
        frame, (1280, 720), cv2.INTER_AREA)], axis=1)

    [hp.draw_text(frame, 16 + 448, 208 + index * 128, (0, 0, 255))
     for index, hp in enumerate(suspects_queue)]

    [hp.draw_text(frame, 16 + 128, 208 + index * 128, (255, 0, 0))
     for index, hp in enumerate(workers_queue)]

    print(colored(time.time() - mid_time, 'blue'))

    cv2.imwrite('./Temp/{}.jpg'.format(time.time(), ), frame)

    # =====================cv_window========================
    # if cv_test_mode:
    #     if windowNotSet is True:
    #         cv2.namedWindow(ip_address, cv2.WINDOW_NORMAL)
    #         cv2.resizeWindow(ip_address, 840, 360)
    #         windowNotSet = False

    #     cv2.imshow(ip_address, frame)
    #     k = cv2.waitKey(1) & 0xff
    #     if k == ord('q') or k == 27:
    #         break
