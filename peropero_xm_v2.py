# coding: utf-8
import cv2
import os
import numpy as np
import time
from termcolor import colored
import asyncio
from multiprocessing import Process, Queue, Manager
import socketio
from helper import read_pkl_model, start_up_init, encode_image
from CXMIPCamera import XMIPCamera
import face_detector
import face_embedding
import functools


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
        # print(mid_time-strat_time, time.time()-mid_time, flush=True)

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
    rate = preload.max_frame_rate
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

        print(colored(loop.time()-start_time, 'red'), flush=True)

        for i in range(int((loop.time() - start_time) * rate + 1)):
            for item in frame_queue.get():
                upstream_queue.put(item)


async def camera_loop(preload):
    reciprocal_of_max_frame_rate = 1/preload.max_frame_rate
    address_dict = preload.address_dict
    camera_dict = {}

    for address in address_dict:
        xmcp = XMIPCamera(address.encode('UTF-8'), 34567, b"admin", b"")
        xmcp.start()
        camera_dict[address] = xmcp

    # =================== ETERNAL LOOP ====================
    loop = asyncio.get_running_loop()
    while True:
        start_time = loop.time()
        frame_queue.put([(address, camera_dict[address].frame(rows=540, cols=960))
                         for address in address_dict])
        restime = reciprocal_of_max_frame_rate - loop.time() + start_time
        if restime > 0:
            await asyncio.sleep(restime)


# =================== ARGS ====================
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
address_dict = ['10.41.0.198', '10.41.0.199']
args = start_up_init()
args.mtcnn_minsize = 288
args.mtcnn_factor = 0.1
args.mtcnn_threshold = [0.92, 0.95, 0.99]
args.address_dict = address_dict

# =================== INIT ====================
frame_queue = Queue(maxsize=args.max_frame_rate)
upstream_queue = Queue(maxsize=args.max_frame_rate * len(address_dict))
suspicion_face_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)

# =================== Process On ====================
Process(target=lambda: asyncio.run(detection_loop(args))).start()
Process(target=lambda: asyncio.run(embedding_loop(args))).start()
Process(target=lambda: asyncio.run(camera_loop(args))).start()
asyncio.run(upload_loop())
