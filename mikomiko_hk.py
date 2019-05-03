# coding: utf-8
import face_embedding
import face_detector
import cv2
import os
import sys
import numpy as np
import time
from termcolor import colored
from helper import read_pkl_model, start_up_init, encode_image
import asyncio
from multiprocessing import Process, Queue, Manager
import socketio
from CHKIPCamera import HKIPCamera


async def upload_loop(url="http://127.0.0.1:6789"):
    # =====================Uploader Setsup========================
    sio = socketio.AsyncClient()
    @sio.on('response', namespace='/remilia')
    async def on_response(data):
        current_address, upload_frame = upstream_frame_queue.get()
        image_string = 0
        # strat_time = time.time()
        if current_address == data:
            image_string = encode_image(upload_frame)
        # mid_time = time.time()
        await sio.emit('frame_data', image_string, namespace='/remilia')
        try:
            img, dt, prob, name = result_queue.get_nowait()
            result_string = {'image': encode_image(img),
                             'time': dt, 'name': name, 'prob': prob}
            await sio.emit('result_data', result_string, namespace='/remilia')
        except Exception as e:
            pass
        # print(mid_time-strat_time, time.time()-mid_time)
        # sys.stdout.flush()

    @sio.on('connect', namespace='/remilia')
    async def on_connect():
        await sio.emit('frame_data', 0, namespace='/remilia')

    await sio.connect(url)
    await sio.wait()


async def embedding_loop(preload):
    # =================== FR MODEL ====================
    mlp, class_names = read_pkl_model('./model-mlp/mlp.pkl')
    preload.gpu = -1
    embedding = face_embedding.EmbeddingModel(preload)
    while True:
        img = suspicion_face_queue.get()
        dt = time.strftime('%m-%d %H:%M:%S')

        predict = mlp.predict_proba([embedding.get_one_feature(img)])
        prob = predict.max(1)[0]

        result_queue.put((img, dt, prob, class_names[predict.argmax(1)[0]]))


async def detection_loop(preload, frame_queue):
    # =================== FD MODEL ====================
    detector = face_detector.DetectorModel(preload)
    fill_number = preload.fill_number
    ip_address = preload.ip_address
    # loop = asyncio.get_running_loop()

    while True:
        # start_time = loop.time()
        # sys.stdout.flush()

        head_frame = frame_queue.get()
        # time.sleep(0.4)

        # tracker = cv2.MultiTracker_create()
        # t_box = []

        for img, box in detector.get_all_boxes(head_frame, save_img=False):
            try:
                if box[4] > 0.98:
                    suspicion_face_queue.put_nowait(img)
            except Exception as e:
                pass

            box = box.astype(int)
            cv2.rectangle(
                head_frame, (box[0], box[1]), (box[2], box[3]), [255, 255, 0], 2)
        # t_box.append(box[:4]/2)

        # print(colored(loop.time()-start_time, 'blue'))
        head_frame = cv2.resize(head_frame, (960, 540), cv2.INTER_AREA)

        # for item in t_box:
        #     tracker.add(cv2.TrackerMedianFlow_create(), head_frame, tuple(item))
        upstream_frame_queue.put((ip_address, head_frame))
        # await sio.emit('frame_data', encode_image(head_frame), namespace='/remilia')

        for i in range(1, fill_number):
            body_frame = frame_queue.get()
            # ok, tricker_boxes = tracker.update(body_frame)
            # if ok:
            #     for box in tricker_boxes:
            #         box = box.astype(int)
            #         cv2.rectangle(body_frame, (box[0], box[1]),
            #                       (box[2], box[3]), [255, 255, 0], 2)
            upstream_frame_queue.put((ip_address, body_frame))
            # await sio.emit('frame_data', encode_image(body_frame), namespace='/remilia')

        # end_time = loop.time()
        # print(colored(loop.time()-track_time, 'red'))
        # sys.stdout.flush()


async def camera_loop(preload):
    video_mode = preload.video_mode
    fill_number = preload.fill_number
    reciprocal_of_max_frame_rate = 1/preload.max_frame_rate

    # =================== VIDEO INTERFACE ====================
    if video_mode:
        video_capture = cv2.VideoCapture("./media/{}".format(ip_address))
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # =================== CAMERA INTERFACE ====================
    else:
        # for address in preload.address_dict:
        #     camera_231 = HKIPCamera(
        #         address.encode('UTF-8'), 8000, b"admin", b"humanmotion01")
        #     camera_231.start()
        address = '10.41.0.231'
        camera_231 = HKIPCamera(address.encode(
            'UTF-8'), 8000, b"admin", b"humanmotion01")
        camera_231.start()
        # camera_232 = HKIPCamera(address.encode(
        #     'UTF-8'), 8000, b"admin", b"humanmotion01")
        # camera_232.start()

    frame_counter = 0
    loop = asyncio.get_running_loop()

    # =================== ETERNAL LOOP ====================
    while True:
        start_time = loop.time()

        # if video_mode:
        #     ok, frame = video_capture.read()
        # else:

        if frame_counter % fill_number:
            frame_queue_231.put(camera_231.frame(rows=540, cols=960))
            # frame_queue_232.put(camera_232.frame(rows=540, cols=960))
        else:
            frame_queue_231.put(camera_231.frame())
            # frame_queue_232.put(camera_232.frame())

        frame_counter = frame_counter % fill_number + 1

        if not frame_counter % 5:
            print(loop.time() - start_time, upstream_frame_queue.qsize(),
                  frame_queue_231.qsize())
            # print(loop.time() - start_time, upstream_frame_queue.qsize(),
            #   frame_queue_231.qsize(), frame_queue_232.qsize())

        restime = reciprocal_of_max_frame_rate - loop.time() + start_time

        if restime > 0:
            await asyncio.sleep(restime)

# =================== INIT ====================
# address_dict = ['10.41.0.231', '10.41.0.232']
address_dict = ['10.41.0.231']
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
frame_buffer_size = 20 * len(address_dict)

frame_queue_231 = Queue(maxsize=frame_buffer_size)
# frame_queue_232 = Queue(maxsize=frame_buffer_size)

upstream_frame_queue = Queue(maxsize=frame_buffer_size)
suspicion_face_queue = Queue(maxsize=frame_buffer_size)
result_queue = Queue(maxsize=frame_buffer_size)

# =================== ARGS ====================
args = start_up_init(train_mode=True)
args.fill_number = 16
args.address_dict = address_dict
# print(args.mtcnn_threshold)

# =================== Process On ====================

# for address in address_dict:
#     args.ip_address = address
#     Process(target=lambda: asyncio.run(detection_loop(args, frame_queue_231))).start()
args.ip_address = '10.41.0.231'
Process(target=lambda: asyncio.run(
    detection_loop(args, frame_queue_231))).start()

# args.ip_address = '10.41.0.232'
# Process(target=lambda: asyncio.run(
#     detection_loop(args, frame_queue_232))).start()

Process(target=lambda: asyncio.run(embedding_loop(args))).start()
Process(target=lambda: asyncio.run(camera_loop(args))).start()
asyncio.run(upload_loop())
