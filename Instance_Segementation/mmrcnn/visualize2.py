# python 2 compability
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from Instance_Segementation.mmrcnn import utils
import os
import sys
import logging
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
import skimage

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

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

class_names_exists = ['BG', 'person', 'bicycle', 'car', 'motorcycle',
               'bus', 'truck', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'cat', 'dog', 'cow', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'laptop']

class_color_dict = {
    'BG': 0,
    'person': 1,
    'bicycle': 2,
    'car': 3,
    'motorcycle': 4,
    'bus': 5,
    'truck': 6,
    'traffic light': 7,
    'fire hydrant': 8,
    'stop sign': 9,
    'parking meter': 10,
    'bench': 11,
    'cat': 12,
    'dog': 13,
    'cow': 14,
    'chair': 15,
    'couch': 16,
    'potted plant': 17,
    'bed': 18,
    'dining table': 19,
    'laptop': 20
}


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    print(colors)
    random.shuffle(colors)
    return colors

def assign_colors(class_names):
    color = random_colors(N = len(class_names), bright = True)
    color_dict={}
    for i,cname in enumerate(class_names):
        color_dict[cname]=color[i]
    print("color dictionary", color_dict)
    return color_dict

color_dict = assign_colors(class_names)

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    mask_px = np.where(mask)
    for c in range(3):
        image[mask_px[0], mask_px[1], c] = color#*10
    return image

def apply_color_mask(image, mask, color, alpha=0.5):
    mask_px = np.where(mask)
    for c in range(3):
        image[mask_px[0], mask_px[1], c] = (1 - alpha)*image[mask_px[0], mask_px[1], c] + alpha * color[c] * 255 #color[c]
    return image



def display_instances_live(image, boxes, masks, class_ids, class_names,
                           scores=None, title="",
                           figsize=(16, 16), ax=None,
                           show_mask=True, show_bbox=False,
                           colors=None, captions=None):
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    colors = []
    colorsRandom = random_colors(N)
    for cname in class_ids:
        print("CName: ", class_names[cname])
        if class_names[cname] in class_names_exists:
            colors.append(class_color_dict[class_names[cname]])
        else:
            colors.append(None)

    height, width = image.shape[:2]

    masked_image = np.zeros((height, width, 3))#image#np.zeros((height, width, 3))
    masked_image_colors = np.zeros((height,width,3)) 
    for i in range(N):
        if colors[i] is not None:
            color = colors[i]
            color2 = colorsRandom[i]
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]

            # Mask
            mask = masks[:, :, i]
            if show_mask:
                masked_image = apply_mask(masked_image, mask, color)
                masked_image_colors = apply_color_mask(masked_image_colors,mask,color2)


    return masked_image.astype(np.uint8),masked_image_colors.astype(np.uint8)
