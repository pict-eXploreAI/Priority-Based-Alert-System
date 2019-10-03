import cv2
import numpy as np
import glob
import time
from scipy.stats import mode
from numpy.lib.stride_tricks import as_strided

# Given 27 classes
# classes_cdac = ['road', 'sidewalk', 'non-drivable', 'person', 'animal', 'rider', 'motorcycle', 'bicycle',
#                 'rickshaw', 'car', 'truck', 'bus', 'train', 'curb', 'wall', 'fence', 'billboard', 'traffic sign',
#                 'traffic light', 'pole', 'poles', 'fallback', 'building', 'bridge', 'vegetation', 'sky',
#                 'background', 'unlabelled']

classes_cdac = ['road', 'sidewalk', 'building', 'wall', 'fence', 'unlabelled']
globals()["alignment"] = __import__("Depth_Model.alignment")


# classes_cdac = ['Sky',
#                 'Building',
#                 'Column-Pole',
#                 'Road',
#                 'Sidewalk',
#                 'Tree',
#                 'Sign-Symbol',
#                 'Fence',
#                 'Car',
#                 'Pedestrain',
#                 'Bicyclist',
#                 'Void']

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle',
               'bus', 'truck', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'cat', 'dog', 'cow', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'laptop']


def relabel(segmentation_mask):
    segmentation_mask[segmentation_mask == 255] = len(classes_cdac)-1
    segmentation_mask[segmentation_mask == 50] = 0
    segmentation_mask[segmentation_mask == 40] = 1
    segmentation_mask[segmentation_mask == 30] = 2
    segmentation_mask[segmentation_mask == 20] = 3
    segmentation_mask[segmentation_mask == 10] = 4
    return segmentation_mask


# Priority wise classes dictionary
weights_class = {'car': 'Automobiles', 'motorcycle': 'Automobiles', 'rickshaw': 'Automobiles',
                 'truck': 'Automobiles', 'bus': 'Automobiles', 'train': 'Automobiles',
                 'person': 'Humans', 'rider': 'Humans', 'bicycle': 'Humans', 'animal': 'Humans',
                 'road': 'Paths', 'sidewalk': 'Paths', 'non-drivable': 'Paths', 'bridge': 'Paths',
                 'curb': 'Obstacles', 'traffic sign': 'Obstacles', 'traffic light': 'Obstacles',
                 'fallback': 'Obstacles', 'billboard': 'Obstacles', 'pole': 'Obstacles',
                 'poles': 'Obstacles', 'vegetation': 'Obstacles', 'wall': 'Walls', 'building': 'Walls',
                 'fence': 'Walls', 'sky': 'Background', 'background': 'background', 'unlabelled': 'background'}

weights_classes_dict = {'Automobiles': ['Car', 'motorcycle', 'rickshaw', 'truck', 'bus', 'train'],
                        'Humans': ['person', 'rider', 'bicycle', 'animal'],
                        'Paths': ['road', 'sidewalk', 'non-drivable', 'bridge'],
                        'Obstacles': ['curb', 'billboard', 'traffic sign', 'traffic light', 'pole', 'poles', 'fallback', 'vegetation'],
                        'Walls': ['wall', 'building', 'fence'],
                        'Background': ['sky', 'background', 'unlabelled']}


# Priority weights list
priority_weights = ['Automobiles', 'Humans',
                    'Paths', 'Obstacles', 'Walls', 'Background']


def get_the_closest(depth_map, mask, mask_colored):
    thresholdSides = 200
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
    depth_map_masked = depth_map * mask_copy
    cv2.imshow("Depth Priority Mask: ", depth_map_masked)

    # central_depth_map_masked = depth_map_masked[:, 104:312]
    # max_element_index = np.unravel_index(np.argmax(central_depth_map_masked, axis=None), central_depth_map_masked.shape)
    # max_element_index = np.unravel_index(np.argmax(depth_map_masked, axis=None), depth_map_masked.shape)
    # object_id = mask[max_element_index[0], max_element_index[1]]
    # print(len(class_names)-1,max_element_index,object_id)
    # object_name = class_names[object_id]

    #closest_message = object_name + " is in front of you."
    leftZ = depth_map_masked[:, :104]
    centerZ = depth_map_masked[:, 104:312]
    rightZ = depth_map_masked[:, 312:]


    leftColor = mask_colored[:, :104]
    centerColor = mask_colored[:, 104:312]
    rightColor = mask_colored[:, 312:]

    # cv2.imshow("Left Depth Priority Mask: ", leftColor)
    # cv2.imshow("Right Depth Priority Mask: ", rightColor)
    # cv2.imshow("Center Depth Priority Mask: ", centerColor)

    max_element_center = np.unravel_index(
        np.argmax(centerZ, axis=None), centerZ.shape)
    max_element_left = np.unravel_index(
        np.argmax(leftZ, axis=None), leftZ.shape)
    max_element_right = np.unravel_index(
        np.argmax(rightZ, axis=None), rightZ.shape)


    print("Max Center: ", max_element_center, centerZ[max_element_center[0],max_element_center[1]])
    print("Max_Left: ", max_element_left)
    print("Max_right: ", max_element_right)

    if centerZ[max_element_center[0],max_element_center[1]] >= thresholdCenter:
        object_id = mask[max_element_center[0], max_element_center[1]+104]
        object_name = class_names[object_id]
        message_list += object_name + " is in front of you. "
        objectColor = mask_colored[max_element_center[0],
                                   max_element_center[1]]
        print('Object Color: ',objectColor)

    # if np.argmax(leftZ, axis=None) >= thresholdSides:
    #     objectColorSide = mask_colored[max_element_left[0],
    #                                    max_element_left[1]]
    #     if objectColor[0] != objectColorSide[0] or objectColor[1] != objectColorSide[1] or objectColor[2] != objectColorSide[2]:
    #         object_id = mask[max_element_left[0], max_element_left[1]]
    #         object_name = class_names[object_id]
    #         message_list += object_name + " is to your left. "

    # if np.argmax(rightZ, axis=None) >= thresholdSides:
    #     objectColorSide = mask_colored[max_element_right[0],
    #                                    max_element_right[1]]
    #     if objectColor[0] != objectColorSide[0] or objectColor[1] != objectColorSide[1] or objectColor[2] != objectColorSide[2]:
    #         object_id = mask[max_element_right[0], max_element_right[1]]
    #         object_name = class_names[object_id]
    #         message_list += object_name + " is to your right. "

    return message_list


def get_the_path_message(segmentation_mask):
    message_list = ""
    segmentation_mask = relabel(segmentation_mask)
    segmentation_mask = cv2.resize(
        segmentation_mask, (416, 128), interpolation=cv2.INTER_NEAREST)
    height, width = segmentation_mask.shape
    start_index, start_column = int(height * 0.7), 0
    lower_part = segmentation_mask[start_index:height, start_column:width]
    left_lower_part = lower_part[:, :104]
    central_lower_part = lower_part[:, 104:312]
    right_lower_part = lower_part[:, 312:]
    walking_on = classes_cdac[mode(central_lower_part, axis=None)[
        0][0]].replace(" ", "")
    left_side = classes_cdac[mode(left_lower_part, axis=None)[
        0][0]].replace(" ", "")
    right_side = classes_cdac[mode(right_lower_part, axis=None)[
        0][0]].replace(" ", "")
    centerVal = mode(segmentation_mask[:, :], axis=None)[0][0]
    center = classes_cdac[centerVal].replace(" ", "")
    print("Walking On: ", walking_on)
    print("Left Side: ", left_side)
    print("Right Side: ", right_side)
    print("Center: ", center, centerVal)
    walking_on_message = "You are currently walking on a " + walking_on + ". "
    message_list += walking_on_message
    if walking_on == "road":
        if left_side == "sidewalk":
            left_side_message = "There is a sidewalk to your left. Please move to your left."
            message_list += left_side_message
        elif right_side == "sidewalk":
            right_side_message = "There is a sidewalk to your right. Please move to your right."
            message_list += right_side_message

    return message_list


def get_the_object_motion_message(object_name, objectmotion_vector):
    objectmotion_vector[0] = objectmotion_vector[0].split(",")
    objectmotion_vector[1] = objectmotion_vector[1].split(",")
    x_diff = float(objectmotion_vector[0][0]) - \
        float(objectmotion_vector[1][0])
    y_diff = float(objectmotion_vector[0][1]) - \
        float(objectmotion_vector[1][1])
    z_diff = float(objectmotion_vector[0][2]) - \
        float(objectmotion_vector[1][2])
    moved = []
    message = object_name
    if abs(x_diff) >= 1:
        if x_diff >= 1:
            moved.append('left')
            message += ' is moving left,'
        else:
            moved.append('right')
            message += ' is moving right,'
    if abs(y_diff) >= 1:
        if y_diff >= 1:
            moved.append('down')
            message += ' is going down the hill,'
        else:
            moved.append('up')
            message += ' is going up the hill,'
    if abs(z_diff) >= 1:
        if z_diff >= 1:
            moved.append('away')
            message += ' is moving away from you.'
        else:
            moved.append('towards')
            message += ' is moving towards you.'
    return message


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
    objectmotion_vector = [[-0.3779029, -0.25309393, 1.4405013, -0.82686526, -0.0763275, 0.14866838],
                           [-0.076262556, -0.29363453, -0.46555933, -0.99811023, -0.4434797, 0.12170079]]
    closest_message = get_the_closest(depth_image, seg_image)
    path_message = get_the_path_message(seg_image)
    object_message = get_the_object_motion_message("car", objectmotion_vector)
    print(closest_message)
    print(path_message)
    print(object_message)
