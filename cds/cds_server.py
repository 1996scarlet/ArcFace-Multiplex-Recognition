import os
import time
import json
import shutil
import threading
import io
import numpy as np
from flask import Flask
from flask import render_template, request, redirect, url_for, send_file
from flask_socketio import SocketIO, send, emit

import memcache
mc = memcache.Client(['127.0.0.1:12000'], debug=True)

app = Flask(__name__)
socketio = SocketIO(app)


@app.route("/lunatic", methods=["GET"])
def lunatic():
    return render_template("lunatic.html")


@socketio.on('connect', namespace='/remiria')
def remiria_connect():
    print('remiria scarlet has been connected')


@socketio.on('change_ip', namespace='/remiria')
def change_current_ip(data):
    mc.set("current_ip", data.get('param'))


@app.route('/upload', methods=['POST'])
def set_base64():
    image_data = request.get_data() or "None"

    socketio.emit('frame_data', {'data': image_data}, namespace='/remiria')

    return "Done"


if __name__ == "__main__":
    socketio.run(app, debug=True, port=6789, host='0.0.0.0')
