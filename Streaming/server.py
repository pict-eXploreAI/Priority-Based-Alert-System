
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

from Instance_Segementation import live_seg  # 2 as live_seg
import time
from ESPNet import test_seg
from Priority import priority
# from text2speech_multilingual import please1

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


import time
@socketio.on('inputImage', namespace='/videoStream')
def videoStream_message(input):
    start = time.time()
    print("Received Frame")
    # output_str = input
    input = input.split(",")[1]

    input_img = base64_to_pil_image(input).convert('RGB')

    cvImg = np.array(input_img)
    # Convert RGB to BGR
    cvImg = cvImg[:, :, ::-1].copy()
    cv2.imshow("Received Frame", cvImg)

    # Semantic Segmentation
    segmented_img = test_seg.evaluate(modelESPNet, cvImg, device)

    # Instance Segmentation
    #instance_img = live_seg.run_instance_model(modelInstance,cvImg)
    instance_img, instance_img_colored = live_seg.run_instance_model(
        modelInstance, cvImg)
    cv2.imshow("Instance Mask", instance_img_colored)

    # Depth and Object Motion
    color_map = depth.depth.run_inference(
        sess=sess, model=inference_model, frame=cvImg)
    color_map = (color_map*255).astype(np.uint8)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    cv2.imshow("Depth", color_map)

    # object motion
    # objectmotion_vector = objectmotion.objectmotion.run_inference(sess=sess, model=inference_model, image_list=[color_map, instance_img])

    # closest_message = priority.get_the_closest(color_map, instance_img)
    # path_message = priority.get_the_path_message(instance_img)
    # objectmotion_message = priority.get_the_object_motion_message("", objectmotion_vector)

    # print("priorityOutput", closest_message[0] + ". " + path_message[0] + ". " + objectmotion_message)

    closest_message = priority.get_the_closest(
        color_map, instance_img, instance_img_colored)
    path_message = priority.get_the_path_message(segmented_img)
    #objectmotion_message = priority.get_the_object_motion_message("", objectmotion_vector)
    final_message = ""

    # + objectmotion_message)
    print("priorityOutput", closest_message + path_message)
    emit("priorityOutput", closest_message + path_message)
    print("Inference time: ", time.time() - start)

    if cv2.waitKey(1) == 27:
        return

    #output_str = "Priority Information"


if __name__ == '__main__':
    host = os.system("hostname -I")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000,keyfile="./key.pem",certfile="./cert.pem",use_reloader=False)
    # socketio.run(app, debug=True, host="192.168.43.165", port=5000,
    #              ssl_context=('./cert.pem', './key.pem'), use_reloader=False)

