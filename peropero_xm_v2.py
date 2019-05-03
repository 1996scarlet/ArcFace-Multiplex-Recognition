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
from CXMIPCamera import XMIPCamera
import math


async def upload_loop(url="http://127.0.0.1:6789"):
    # =====================Uploader Setsup========================
    sio = socketio.AsyncClient()
    @sio.on('response', namespace='/flandre')
    async def on_response(data):
        current_address, upload_frame = upstream_queue.get()
        image_string = 0
        # strat_time = time.time()
        if current_address == data:
            image_string = encode_image(upload_frame)
        # mid_time = time.time()
        await sio.emit('frame_data', image_string, namespace='/flandre')
        try:
            ip, img, dt, prob, name = result_queue.get_nowait()
            result_string = {'image': encode_image(img),
                             'time': dt, 'name': name, 'prob': prob, 'ip': ip}
            await sio.emit('result_data', result_string, namespace='/flandre')
        except Exception as e:
            pass
        # print(mid_time-strat_time, time.time()-mid_time)
        # sys.stdout.flush()

    @sio.on('connect', namespace='/flandre')
    async def on_connect():
        await sio.emit('frame_data', 0, namespace='/flandre')

    await sio.connect(url)
    await sio.wait()


async def embedding_loop(preload):
    # =================== FR MODEL ====================
    mlp, class_names = read_pkl_model('./model-mlp/mlp.pkl')
    embedding = face_embedding.EmbeddingModel(preload)
    while True:
        ip, img = suspicion_face_queue.get()
        dt = time.strftime('%m-%d %H:%M:%S')
        predict = mlp.predict_proba([embedding.get_one_feature(img)])
        prob = predict.max(1)[0]
        name = class_names[predict.argmax(1)[0]]
        result_queue.put((ip, img, dt, prob, name))


async def detection_loop(preload):
    # =================== FD MODEL ====================
    detector = face_detector.DetectorModel(preload)
    loop = asyncio.get_running_loop()

    while True:
        start_time = loop.time()
        head_frame_list = frame_queue.get()

        for (ip_address, head_frame) in head_frame_list:
            for img, box in detector.get_all_boxes(head_frame, save_img=False):
                try:
                    suspicion_face_queue.put_nowait((ip_address, img))
                except Exception as e:
                    pass

                box = box.astype(int)
                cv2.rectangle(
                    head_frame, (box[0], box[1]), (box[2], box[3]), [255, 255, 0], 2)

            upstream_queue.put((ip_address, head_frame))

        print(colored(loop.time()-start_time, 'red'))
        fill_number = math.floor((loop.time() - start_time) * 25)

        for i in range(0, fill_number):
            for item in frame_queue.get():
                upstream_queue.put(item)

        sys.stdout.flush()


async def camera_loop(preload):
    reciprocal_of_max_frame_rate = 1/preload.max_frame_rate
    address_dict = preload.address_dict
    camera_dict = {}

    for address in address_dict:
        xmcp = XMIPCamera(address.encode('UTF-8'), 34567, b"admin", b"")
        xmcp.start()
        camera_dict[address] = xmcp

    frame_counter = 0
    loop = asyncio.get_running_loop()

    # =================== ETERNAL LOOP ====================
    while True:
        start_time = loop.time()
        frame_queue.put([(address, camera_dict[address].frame(rows=540, cols=960))
                         for address in address_dict])

        # frame_counter = frame_counter % 1000
        # if not frame_counter % 5:
        #     print(upstream_queue.qsize(), frame_queue.qsize())

        restime = reciprocal_of_max_frame_rate - loop.time() + start_time
        if restime > 0:
            await asyncio.sleep(restime)

# =================== INIT ====================
address_dict = ['10.41.0.198', '10.41.0.199']
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
frame_buffer_size = 10 * len(address_dict)

frame_queue = Queue(maxsize=frame_buffer_size)
upstream_queue = Queue(maxsize=frame_buffer_size)
suspicion_face_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)

# =================== ARGS ====================
args = start_up_init(train_mode=True)
args.mtcnn_minsize = 288
args.mtcnn_factor = 0.1
args.mtcnn_threshold = [0.92, 0.95, 0.99]
args.address_dict = address_dict

# =================== Process On ====================
Process(target=lambda: asyncio.run(detection_loop(args))).start()
Process(target=lambda: asyncio.run(embedding_loop(args))).start()
Process(target=lambda: asyncio.run(camera_loop(args))).start()
asyncio.run(upload_loop())
