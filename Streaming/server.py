""" The Flask Server for the PWA """

import sys
import os
import requests
import cv2
import socket
import numpy as np
import logging
from sys import stdout
from utils import *
from camera import Camera

from flask import Flask, session, render_template, request, url_for, flash, redirect, jsonify, Response
from flask_socketio import SocketIO, emit
sys.path.append("../")
sys.path.append("./")

from Instance_Segementation import live_seg
import time
from ESPNet import test_seg
from Priority import priority

# Initialize Flask
app = Flask(__name__)
app.config["SECRET_KEY"] = 'ks2334'
socketio = SocketIO(app)

# Load ESPNet
modelESPNet, device = test_seg.load_model()

# Load MaskRCNN
modelInstance = live_seg.load_instance_model()

# Load Depth
depth = None
globals()["depth"] = __import__("Depth_Model.depth")
globals()["objectmotion"] = __import__("Depth_Model.objectmotion")

# Load Depth Model
sess, inference_model = depth.depth.load_depth_model(
    "../Trained-Models/depth-model/model-199160")


@app.route("/")
def index():
    return render_template('index.html')


@socketio.on('connect', namespace='/videoStream')
def videoStream_connect():
    print("Client Connected")


# Socket code for connecting to the client for video streaming from client-server
import time
@socketio.on('inputImage', namespace='/videoStream')
def videoStream_message(input):
    start = time.time()
    print("Received Frame")
    input = input.split(",")[1]

    # Decoding the base64 image
    input_img = base64_to_pil_image(input).convert('RGB')

    cvImg = np.array(input_img)
    # Convert RGB to BGR
    cvImg = cvImg[:, :, ::-1].copy()
    cv2.imshow("Received Frame", cvImg)

    # Semantic Segmentation
    segmented_img = test_seg.evaluate(modelESPNet, cvImg, device)

    # Instance Segmentation
    instance_img, instance_img_colored = live_seg.run_instance_model(
        modelInstance, cvImg)
    cv2.imshow("Instance Mask", instance_img_colored)

    # Depth and Object Motion
    color_map = depth.depth.run_inference(
        sess=sess, model=inference_model, frame=cvImg)
    color_map = (color_map*255).astype(np.uint8)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    cv2.imshow("Depth", color_map)

    closest_message = priority.get_the_closest(
        color_map, instance_img, instance_img_colored)
    path_message = priority.get_the_path_message(segmented_img)
    final_message = ""

    print("priorityOutput", closest_message + path_message)
    emit("priorityOutput", closest_message + path_message)
    print("Inference time:", time.time() - start)

    if cv2.waitKey(1) == 27:
        return


if __name__ == '__main__':
    host = os.system("hostname -I")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000, keyfile="./key.pem", certfile="./cert.pem", use_reloader=False)
    # Code for lower Django version, if you get error for keyfile and certfile syntac
    # socketio.run(app, debug=True, host="192.168.43.165", port=5000,
    #              ssl_context=('./cert.pem', './key.pem'), use_reloader=False)

