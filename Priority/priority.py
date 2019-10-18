import cv2
import numpy as np
import glob
import time
from scipy.stats import mode
from numpy.lib.stride_tricks import as_strided

# Classes for semantic segmentation
classes_cdac = ['road', 'sidewalk', 'building', 'wall', 'fence', 'unlabelled']
globals()["alignment"] = __import__("Depth_Model.alignment")

# Classes for instance segmentation
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle',
               'bus', 'truck', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'cat', 'dog', 'cow', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'laptop']

# Relabelling the segmentation mask
def relabel(segmentation_mask):
    segmentation_mask[segmentation_mask == 255] = len(classes_cdac)-1
    segmentation_mask[segmentation_mask == 50] = 0
    segmentation_mask[segmentation_mask == 40] = 1
    segmentation_mask[segmentation_mask == 30] = 2
    segmentation_mask[segmentation_mask == 20] = 3
    segmentation_mask[segmentation_mask == 10] = 4
    return segmentation_mask


def get_the_closest(depth_map, mask, mask_colored):
    '''
        Input: depth_map - The depth map obtained by the depth model
               mask - The Instance Segmented mask in gray-scale (Similar objects will have same color eg. two cars are of same color)
               mask_colored - The Original Instance Segmented mask in color-channels (Each identified object will be of different color)
        Output: It returns the message_list i.e. the list of message telling the closest objects
        Description:
            We first resize both depth_map and mask to 416*128 shape, then we will divide the image in 3 parts left, center and right part.
            Then we will retrive the most closest objects in each part using depth map, and identifies the object using instance mask. And 
            we will construct the message containing that part name and identified object.
    '''
    thresholdSides = 10
    thresholdCenter = 100
    message_list = ""
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    print("Mask: ", np.unique(mask), mask.shape)
    mask_copy = mask.copy()
    depth_map = cv2.resize(depth_map, (416, 128),
                           interpolation=cv2.INTER_NEAREST)
    mask_copy = cv2.resize(mask_copy, (416, 128),
                           interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (416, 128), interpolation=cv2.INTER_NEAREST)
    mask_colored = cv2.resize(mask_colored, (416, 128),
                              interpolation=cv2.INTER_NEAREST)
    mask_copy[mask_copy > 0] = 1
    mask_copy[mask_copy == 0] = 0
    depth_map_masked = depth_map * mask_copy
    cv2.imshow("Depth Priority Mask: ", depth_map_masked)

    leftZ = depth_map_masked[:, :104]
    centerZ = depth_map_masked[:, 104:312]
    rightZ = depth_map_masked[:, 312:]

    leftColor = mask_colored[:, :104]
    centerColor = mask_colored[:, 104:312]
    rightColor = mask_colored[:, 312:]

    max_element_center = np.unravel_index(
        np.argmax(centerZ, axis=None), centerZ.shape)
    max_element_left = np.unravel_index(
        np.argmax(leftZ, axis=None), leftZ.shape)
    max_element_right = np.unravel_index(
        np.argmax(rightZ, axis=None), rightZ.shape)

    # if centerZ[max_element_center[0],max_element_center[1]] >= thresholdCenter:
    object_id = mask[max_element_center[0], max_element_center[1]+104]
    if object_id != 0:
        object_name = class_names[object_id]
        if object_name == 'truck':
            object_name = 'car'
        if object_name.replace(" ", '') != 'BG' or object_name.replace(" ", '') != 'sky':
            message_list += object_name + " is in front of you."
        objectColor = mask_colored[max_element_center[0],
                                    max_element_center[1]+104]
        
        #print('Object Color: ',objectColor)
        
    if leftZ[max_element_left[0],max_element_left[1]] >= thresholdSides:
        objectColorSide = mask_colored[max_element_left[0],
                                    max_element_left[1]]
        if objectColor[0] != objectColorSide[0] or objectColor[1] != objectColorSide[1] or objectColor[2] != objectColorSide[2]:
            object_id = mask[max_element_left[0], max_element_left[1]]
            object_name = class_names[object_id]
            if object_name == 'truck':
                object_name = 'car'
            print(object_name)
            if object_name != 'BG' or object_name != 'sky':
                message_list += object_name + " is to your left. "

    if rightZ[max_element_right[0],max_element_right[1]] >= thresholdSides:
        objectColorSide = mask_colored[max_element_right[0],
                                    max_element_right[1]+312]
        if objectColor[0] != objectColorSide[0] or objectColor[1] != objectColorSide[1] or objectColor[2] != objectColorSide[2]:
            object_id = mask[max_element_right[0], max_element_right[1]+312]
            object_name = class_names[object_id]
            if object_name == 'truck':
                object_name = 'car'
            print(object_name)
            if object_name != 'BG' or object_name != 'sky':
                message_list += object_name + " is to your right. "

    return message_list


def get_the_path_message(segmentation_mask):
    '''
        Input: segmentation_mask - This is segmentation mask obtained from the segmentation model, which contains above mentioned classes
        Output: It returns the message_list i.e. the list of message telling where you are currently walking on and the nearest sidewalk
        Description:
            Similar to the closest_message function, here also we divide the image in left-right parts and check for the sidewalk in both
            parts. And we also check for the most occuring path (road, unlabelled pavement, sidewalk) in the center. And we frame the
            message according to it.
    '''
    message_list = ""
    segmentation_mask = relabel(segmentation_mask)
    segmentation_mask = cv2.resize(
        segmentation_mask, (416, 128), interpolation=cv2.INTER_NEAREST)
    height, width = segmentation_mask.shape
    start_index, start_column = int(height * 0.7), 0
    lower_part = segmentation_mask[:, start_column:width]
    left_lower_part = lower_part[:, :104]
    central_lower_part = lower_part[:, 104:312]
    right_lower_part = lower_part[:, 312:]
    walking_on = classes_cdac[mode(central_lower_part, axis=None)[0][0]].replace(" ", "")
    left_side = ''
    right_side = ''
    if 1 in np.unique(left_lower_part) or 5 in np.unique(left_lower_part):
        left_side = 'sidewalk'
    if 1 in np.unique(right_lower_part) or 5 in np.unique(right_lower_part):
        right_side = 'sidewalk'
    centerVal = mode(segmentation_mask[:, :], axis=None)[0][0]
    center = classes_cdac[centerVal].replace(" ", "")
    print("Walking On: ", walking_on)
    print("Left Side: ", left_side)
    print("Right Side: ", right_side)
    print("Center: ", center, centerVal)
    if walking_on == 'unlabelled':
         walking_on = 'pavment'
    walking_on_message = "You are currently walking on a " + walking_on + ". "
    message_list += walking_on_message
    # if walking_on == "road":
    if left_side == "sidewalk":
        left_side_message = "There is a sidewalk to your left. Please move to your left."
        message_list += left_side_message
    elif right_side == "sidewalk":
        right_side_message = "There is a sidewalk to your right. Please move to your right."
        message_list += right_side_message

    return message_list


if __name__ == "__main__":
    # load_images()
    depth_image_dir = "./depth/"
    image_depth_name = "928063_leftImg8bit.png"
    image_depth = cv2.imread(depth_image_dir + image_depth_name)
    depth_image = cv2.cvtColor(image_depth, cv2.COLOR_BGR2GRAY)
    depth_image = np.divide(depth_image, 255.0)
    segmentation_mask_dir = "./segment/"
    image_seg_name = "928063_leftImg8bit.png"
    image_seg = cv2.imread(segmentation_mask_dir + image_seg_name)
    image_seg = cv2.resize(image_seg, (416, 128))
    seg_image = cv2.cvtColor(image_seg, cv2.COLOR_BGR2GRAY)
    
    closest_message = get_the_closest(depth_image, seg_image)
    path_message = get_the_path_message(seg_image)
    print(closest_message)
    print(path_message)
