from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append("../")
sys.path.append("./")  # To find local version of the library
import random
import math
import re
import datetime
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io
import Instance_Segementation.coco as coco
import cv2
import time

from Instance_Segementation.mmrcnn import utils
from Instance_Segementation.mmrcnn import visualize
from Instance_Segementation.mmrcnn.visualize import display_images
import Instance_Segementation.mmrcnn.model as modellib
from Instance_Segementation.mmrcnn.model import log

ROOT_DIR = os.path.abspath("../")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
DEFAULT_WEIGHTS = os.path.join(ROOT_DIR, "Trained-Models/instance-segmentation-model/mobile_mask_rcnn_coco.h5")


config = coco.CocoConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    POST_NMS_ROIS_INFERENCE = 100

config = InferenceConfig()
#DEVICE = "/cpu:0"
DEVICE = "/gpu:0"

# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']

class_names_exist = ['BG', 'person', 'bicycle', 'car', 'motorcycle',
               'bus', 'truck', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'cat', 'dog', 'cow', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'laptop']


def load_instance_model():
    # Create model in inference mode
    TEST_MODE = "inference"
    #TEST_MODE = "training"
    model = modellib.MaskRCNN(mode=TEST_MODE, model_dir=MODEL_DIR,config=config)
    # Set path to model weights
    weights_path = DEFAULT_WEIGHTS
    # Load weights
    print("Loading Mask RCNN weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    return model

def run_instance_model(model,frame):
    results = model.detect([frame], verbose=1)
    r = results[0]
    print(r['class_ids'])
    frame = visualize.display_instances_live(frame, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'],show_mask=True)
    
    return frame



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    model = load_instance_model()
    while(True):
        start = time.time()
        ret, frame = cap.read()
        frame = run_instance_model(model,frame)
        cv2.imshow('frame',frame)
        print("Time to Process Frames: ",time.time()-start)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

