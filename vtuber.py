# coding: utf-8
import cv2
import os
import numpy as np
import time
from termcolor import colored
import asyncio
from multiprocessing import Process, Queue
import socketio
from helper import start_up_init, encode_image, get_head_pose, line_pairs, eye_aspect_ratio
import IPCamera.interface as ipc
import face_detector
import mss

# =====================DLIB UTILS========================
import dlib
from imutils import face_utils


async def upload_loop(url="http://127.0.0.1:6789"):
    # =====================Uploader Setsup========================
    sio = socketio.AsyncClient()
    @sio.on('response', namespace='/sakuya')
    async def on_response(data):
        upload_frame = upstream_queue.get()
        await sio.emit('frame_data', encode_image(upload_frame), namespace='/sakuya')
        try:
            euler_angle, shape, leftEAR, rightEAR = result_queue.get_nowait()
            result_string = {'X': euler_angle[0, 0], 'Y': euler_angle[1, 0], 'Z': euler_angle[2, 0],
                             'shape': shape.tolist(), 'leftEAR': leftEAR, 'rightEAR': rightEAR}
            await sio.emit('result_data', result_string, namespace='/sakuya')
        except Exception as e:
            print(e)
            pass

    @sio.on('connect', namespace='/sakuya')
    async def on_connect():
        await sio.emit('frame_data', 0, namespace='/sakuya')

    await sio.connect(url)
    await sio.wait()


async def camera_loop(preload):
    # =================== FD MODEL ====================
    rate = preload.max_frame_rate
    # reciprocal_of_max_frame_rate = 1/preload.max_frame_rate
    reciprocal_of_max_frame_rate = 1/60
    loop = asyncio.get_running_loop()

    with mss.mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        while True:
            start_time = loop.time()

            img = np.array(sct.grab(monitor))
            upstream_queue.put(img)

            restime = reciprocal_of_max_frame_rate - loop.time() + start_time
            if restime > 0:
                await asyncio.sleep(restime)


async def detection_loop(preload):
    reciprocal_of_max_frame_rate = 1/preload.max_frame_rate
    address = preload.address
    xmcp = ipc.XMIPCamera(address.encode('UTF-8'), 34567, b"admin", b"")
    xmcp.start()

    # =================== FD MODEL ====================
    detector = face_detector.DetectorModel(preload)
    rate = preload.max_frame_rate
    loop = asyncio.get_running_loop()

    predictor = dlib.shape_predictor(preload.face_landmark_path)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # =================== ETERNAL LOOP ====================
    while True:
        start_time = loop.time()

        try:
            head_frame = xmcp.frame(rows=540, cols=960)

            for img, box, points in detector.get_boxes_and_landmarks(head_frame):
                box = box.astype(int)
                # nimg = cv2.resize(
                # head_frame[box[1]:box[3], box[0]:box[2]], (112, 112))
                shape = predictor(img, dlib.rectangle(0, 0, 112, 112))
                # shape = predictor(head_frame, dlib.rectangle(
                #     box[0], box[1], box[2], box[3]))
                # shape = face_utils.shape_to_np(shape) + 50
                shape = face_utils.shape_to_np(shape)
                reprojectdst, euler_angle = get_head_pose(shape)

                # for (x, y) in shape:
                #     cv2.circle(head_frame, (x, y), 2, (0, 0, 255), -1)

                # for start, end in line_pairs:
                #     cv2.line(
                #         head_frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                # =================== EYES ====================
                shape = shape * 10

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # head_frame = np.zeros([540, 960, 3])

                result_queue.put((euler_angle, shape, leftEAR, rightEAR))
        except Exception as e:
            print(e)
            pass

        print(colored(loop.time()-start_time, 'red'), flush=True)

        restime = reciprocal_of_max_frame_rate - loop.time() + start_time
        if restime > 0:
            await asyncio.sleep(restime)


# =================== ARGS ====================
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
args = start_up_init()
args.mtcnn_minsize = 168
args.mtcnn_factor = 0.1
args.mtcnn_threshold = [0.5, 0.6, 0.7]
args.address = '10.41.0.198'
args.face_landmark_path = './model/landmarks.dat'

# =================== INIT ====================
frame_queue = Queue(maxsize=args.max_frame_rate)
upstream_queue = Queue(maxsize=args.max_frame_rate)
result_queue = Queue(maxsize=args.max_frame_rate)

# =================== Process On ====================
Process(target=lambda: asyncio.run(detection_loop(args))).start()
Process(target=lambda: asyncio.run(camera_loop(args))).start()
asyncio.run(upload_loop())
