# coding: utf-8
import cv2
import os
import numpy as np
import time
from termcolor import colored
from helper import read_pkl_model, start_up_init, encode_image
from multiprocessing import Process, Queue
import asyncio
import socketio
import IPCamera.interface as ipc
import face_embedding
import face_detector


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
            result_string = {
                'image': encode_image(img),
                'time': dt,
                'name': name,
                'prob': prob
            }
            await sio.emit('result_data', result_string, namespace='/remilia')
        except Exception as e:
            pass
        # print(mid_time-strat_time, time.time()-mid_time)

    @sio.on('connect', namespace='/remilia')
    async def on_connect():
        await sio.emit('frame_data', 0, namespace='/remilia')

    await sio.connect(url)
    await sio.wait()


async def embedding_loop(preload):
    # =================== FR MODEL ====================
    mlp, class_names = read_pkl_model(preload.classification)
    embedding = face_embedding.EmbeddingModel(preload)
    while True:
        img = suspicion_face_queue.get()
        dt = time.strftime('%m-%d %H:%M:%S')
        predict = mlp.predict_proba([embedding.get_one_feature(img)])
        prob = predict.max(1)[0]
        name = class_names[predict.argmax(1)[0]]
        result_queue.put((img, dt, prob, name))
    # [[0.30044544 0.31831665 0.30363247 0.07760544]]


async def detection_loop(preload, frame_queue):
    # =================== FD MODEL ====================
    detector = face_detector.DetectorModel(preload)
    ip_address = preload.ip_address
    embedding_threshold = preload.embedding_threshold
    loop = asyncio.get_running_loop()

    while True:
        start_time = loop.time()
        head_frame = frame_queue.get()

        # tracker = cv2.MultiTracker_create()
        # t_box = []
        for img, box in detector.get_all_boxes(head_frame, save_img=False):
            if box[4] > embedding_threshold:
                try:
                    suspicion_face_queue.put_nowait(img)
                except Exception as _:
                    pass

            box = box.astype(np.int)
            cv2.rectangle(head_frame, (box[0], box[1]), (box[2], box[3]),
                          [255, 255, 0], 2)
        # t_box.append(box[:4]/2)

        # print(colored(loop.time() - start_time, 'blue'))
        # head_frame = cv2.resize(head_frame, (960, 540), cv2.INTER_AREA)

        # for item in t_box:
        #     tracker.add(cv2.TrackerMedianFlow_create(), head_frame, tuple(item))
        upstream_frame_queue.put((ip_address, head_frame))
        print(colored(loop.time() - start_time, 'red'), flush=True)

        for i in range(int((loop.time() - start_time) * 25)):
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


async def camera_loop(preload):
    reciprocal_of_max_frame_rate = 1 / preload.max_frame_rate
    address_dict = preload.address_dict
    camera_dict = {}

    # from CXMIPCamera import XMIPCamera
    # for address in address_dict:
    #     xmcp = XMIPCamera(address.encode('UTF-8'), 34567, b"admin", b"")
    #     xmcp.start()
    #     camera_dict[address] = xmcp

    for address in address_dict:
        hkcp = ipc.HKIPCamera(address.encode('UTF-8'), 8000, b"admin",
                              b"humanmotion01")
        hkcp.start()
        camera_dict[address] = hkcp

    frame_counter = 0
    loop = asyncio.get_running_loop()

    # =================== ETERNAL LOOP ====================
    while True:
        start_time = loop.time()
        frame_queue_231.put(camera_dict['10.41.0.231'].frame(rows=540,
                                                             cols=960))
        # frame_queue_231.put(camera_dict['10.41.0.198'].frame(rows=540, cols=960))
        # frame_queue_232.put(camera_dict['10.41.0.199'].frame(rows=540, cols=960))

        # frame_counter = frame_counter % 1000
        # if not frame_counter % 5:
        #     print(loop.time() - start_time, upstream_frame_queue.qsize(),
        #           frame_queue_231.qsize())
        # print(loop.time() - start_time, upstream_frame_queue.qsize(),
        #   frame_queue_231.qsize(), frame_queue_232.qsize())

        restime = reciprocal_of_max_frame_rate - loop.time() + start_time
        if restime > 0:
            await asyncio.sleep(restime)


# =================== INIT ====================
# address_dict = ['10.41.0.198', '10.41.0.199']
address_dict = ['10.41.0.231']
# frame_queue_232 = Queue(maxsize=frame_buffer_size)
# Process(target=lambda: asyncio.run(
#     detection_loop(args, frame_queue_232))).start()
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
frame_buffer_size = 25 * len(address_dict)
upstream_frame_queue = Queue(maxsize=frame_buffer_size)
suspicion_face_queue = Queue(maxsize=frame_buffer_size)
result_queue = Queue(maxsize=frame_buffer_size)

# =================== ARGS ====================
args = start_up_init()
args.address_dict = address_dict

# =================== Process On ====================
args.ip_address = '10.41.0.231'
frame_queue_231 = Queue(maxsize=frame_buffer_size)
Process(
    target=lambda: asyncio.run(detection_loop(args, frame_queue_231))).start()

# args.ip_address = '10.41.0.232'
# frame_queue_232 = Queue(maxsize=frame_buffer_size)
# Process(target=lambda: asyncio.run(
#     detection_loop(args, frame_queue_232))).start()

Process(target=lambda: asyncio.run(embedding_loop(args))).start()
Process(target=lambda: asyncio.run(camera_loop(args))).start()
asyncio.run(upload_loop())
