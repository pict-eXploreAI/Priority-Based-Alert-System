import sys
sys.path.append("../")
sys.path.append("./")
import torch
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
from ESPNet.utilities.print_utils import *
from ESPNet.transforms.classification.data_transforms import MEAN, STD
from ESPNet.utilities.utils import model_parameters, compute_flops

from ESPNet.commons.general_details import segmentation_models, segmentation_datasets
from ESPNet.model.weight_locations.segmentation import model_weight_map

CITYSCAPE_CLASS_LIST = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle', 'background']
image_list = []

class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]

args = AttrDict({'model': 'espnetv2', 'weights_test': '', 's': 2.0, 'data_path': '',
            'dataset': 'city', 'im_size': [512, 256], 'split': 'val', 'model_width': 224,
            'channels': 3, 'num_classes': 1000})

model_key = '{}_{}'.format(args.model, args.s)
dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size[0], args.im_size[1])


args.weights = ''
args.weights_test = "../ESPNet/" + model_weight_map[model_key][dataset_key]['weights']

def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img >= 5] = 255
    img[img == 4] = 10
    img[img == 3] = 20
    img[img == 2] = 30
    img[img == 1] = 40
    img[img == 0] = 50
    return img


def data_transform(img, im_size):
    img = cv2.resize(img, im_size, Image.BILINEAR)
    img = F.to_tensor(img)  # convert to tensor (values between 0 and 1)
    img = F.normalize(img, MEAN, STD)  # normalize the tensor
    return img


def evaluate(model, img, device):
    im_size = (512, 256)
    
    model.eval()
    h, w, _ = img.shape
    img = data_transform(img, im_size)
    img = img.unsqueeze(0)  # add a batch dimension
    img = img.to(device)
    img_out = model(img)
    img_out = img_out.squeeze(0)  # remove the batch dimension
    img_out = img_out.max(0)[1].byte()  # get the label map
    img_out = img_out.to(device='cpu').numpy()
    
    img_out = Image.fromarray(img_out)
    # resize to original size
    img_out = np.array(img_out.resize((w, h), Image.NEAREST))
    im_final = relabel(img_out)
    cv2.imshow("ESPNet", im_final)
    return im_final


def main(model, device):
    # cv2.namedWindow('Recording', cv2.WINDOW_AUTOSIZE)
    # ret,img=cap.read()
    cap = cv2.VideoCapture(0)
    import time
    while True:
        start = time.time()
        ret, img = cap.read()
        evaluate(model, img, device=device)
        print("Time inference:", time.time() - start)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    

def load_model():
    from ESPNet.model.segmentation.espnetv2 import espnetv2_seg
    seg_classes = len(CITYSCAPE_CLASS_LIST)
    args.classes = seg_classes
    model = espnetv2_seg(args)
    num_params = model_parameters(model)
    if args.weights_test:
        print_info_message('Loading model weights')
        weight_dict = torch.load(
            args.weights_test, map_location=torch.device('cuda'))
        model.load_state_dict(weight_dict)
        print_info_message('Weight loaded successfully')
    else:
        print_error_message(
            'weight file does not exist or not specified. Please check: {}', format(args.weights_test))

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    model = model.to(device=device)
    return model, device


if __name__ == '__main__':
    model, device = load_model()

    main(model, device)
