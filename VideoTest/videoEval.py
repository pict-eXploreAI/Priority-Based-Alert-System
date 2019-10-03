# Video file name
video_file = "trim.mov"

import sys
import os
import requests
import cv2
import socket
import numpy as np
import logging
from sys import stdout
sys.path.append("../")
sys.path.append("./")

from Instance_Segementation import live_seg #2 as live_seg 
from Priority import priority
from ESPNet import test_seg
import time
from text2speech_multilingual import please1

# Load ESPNet
modelESPNet, device = test_seg.load_model()

# Load MaskRCNN
modelInstance = live_seg.load_instance_model()

# Load Depth
depth = None
globals()["depth"] = __import__("Depth_Model.depth")
globals()["objectmotion"] = __import__("Depth_Model.objectmotion")
globals()["alignment"] = __import__("Depth_Model.alignment")


# Load Depth Model
sess, inference_model = depth.depth.load_depth_model(
    "../Trained-Models/depth-model/model-199160")


cap = cv2.VideoCapture(video_file)

# Give Message
import threading
class mythread(threading.Thread):
    def __init__(self, msg):
        threading.Thread.__init__(self)
        self.msg = msg

    def run(self):
        print("Giving Message")
        please1.give_message(self.msg)
        time.sleep(7)


cnt = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if cnt>600:
        start = time.time()
        #ret, frame = cap.read()
        #cv2.putText(frame,"Hello World!!!", (0,0), cv2.FONT_HERSHEY_SIMPLEX, 1000, (255,0,0),10)
        #cv2.imshow('frame', frame)

        frame = cv2.resize(frame,(640,480))
        cv2.imwrite(str(cnt)+".png", frame)
        #Semantic Segmentation
        segmented_img = test_seg.evaluate(modelESPNet,frame,device) 
        cv2.imwrite(str(cnt)+"-seg.png", segmented_img)
        #Instance Segmentation
        instance_img,instance_img_colored = live_seg.run_instance_model(modelInstance,frame)
        cv2.imshow("Instance Mask",instance_img_colored)
        cv2.imwrite(str(cnt)+"-inst.png", instance_img_colored)
        #Depth and Object Motion
        color_map = depth.depth.run_inference(sess=sess, model=inference_model, frame=frame)
        color_map = (color_map*255).astype(np.uint8)
        color_map = cv2.cvtColor(color_map,cv2.COLOR_RGB2BGR)
        cv2.imshow("Depth",color_map)
        cv2.imwrite(str(cnt)+"-depth.png", color_map)
        # object motion
        #objectmotion_vector = objectmotion.objectmotion.run_inference(sess=sess, model=inference_model, image_list=[color_map, instance_img])

        closest_message = priority.get_the_closest(color_map, instance_img, instance_img_colored)
        path_message = priority.get_the_path_message(segmented_img)
        #objectmotion_message = priority.get_the_object_motion_message("", objectmotion_vector)
        final_message = ""

        print("priorityOutput", closest_message + path_message )#+ objectmotion_message)
        if cnt%20 == 0:
            # please1.give_message(closest_message + path_message)
            thread1 = mythread(closest_message + path_message)
            thread1.start()
            # time.sleep(2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print("Time Per Frame: ",time.time()-start)
        #time.sleep(0.2)
    cnt+=1
    print("Frame No: ",cnt)

cap.release()
cv2.destroyAllWindows()

