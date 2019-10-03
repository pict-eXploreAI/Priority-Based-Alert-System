"""Runs struct2depth at inference. Produces depth estimates, ego-motion and object motion."""
import cv2
import functools
import sys
# sys.path.append("./")
sys.path.append("../")
from Depth_Model.util import normalize_depth_for_display, get_vars_to_save_and_restore
import tensorflow as tf
import numpy as np
from Depth_Model.model import Model
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# load the tensoflow file
gfile = tf.gfile


def load_depth_model(path):
    # model_ckpt is the location of the stored weights for struct2depth model
    model_ckpt=path
    # loads the pretrained-model
    inference_model = Model(batch_size=1,
                            img_height=128,
                            img_width=416,
                            seq_length=3,
                            architecture='resnet',
                            imagenet_norm=True,
                            use_skip=True,
                            joint_encoder=True)
    # loads the variables of the model_ckpt in a variable
    vars_to_restore = get_vars_to_save_and_restore(model_ckpt)
    # creates the saver to save the variables loaded
    saver = tf.train.Saver(vars_to_restore)
    # creating session
    sess = tf.Session()
    # loads the model into session
    saver.restore(sess, model_ckpt)
    return sess,inference_model


def run_inference(model=None,
                   image_list=None,
                   sess=None,
                   img_width=416,
                   img_height=128):
    """Runs inference. Refer to flags in inference.py for details."""
    img = image_list[0]
    img_seg = image_list[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width * 3, img_height))
    img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)
    img_seg = cv2.resize(img_seg, (img_width * 3, img_height), interpolation=cv2.INTER_NEAREST)


    input_image_stack = np.concatenate(
        [img[:, :img_width], img[:, img_width:img_width*2],
            img[:, img_width*2:]], axis=2)
    input_image_stack = np.expand_dims(input_image_stack, axis=0)

    input_seg_seq = [img_seg[:, :img_width], img_seg[:, img_width:img_width*2],
                     img_seg[:, img_width*2:]]

    input_image_stack = mask_image_stack(input_image_stack,
                                         input_seg_seq)
    
    est_objectmotion = model.inference_objectmotion(
        input_image_stack, sess)
    print("EST_objectmotion:", est_objectmotion)
    est_objectmotion = np.squeeze(est_objectmotion)
    objectmotion_1_2 = ','.join([str(d) for d in est_objectmotion[0]])
    objectmotion_2_3 = ','.join([str(d) for d in est_objectmotion[1]])
    print("ObjectMotion1_2:", objectmotion_1_2)
    print("ObjectMotion2_3:", objectmotion_2_3)

    return [objectmotion_1_2, objectmotion_2_3]


def mask_image_stack(input_image_stack, input_seg_seq):
    """Masks out moving image contents by using the segmentation masks provided.

    This can lead to better odometry accuracy for motion models, but is optional
    to use. Is only called if use_masks is enabled.
    Args:
      input_image_stack: The input image stack of shape (1, H, W, seq_length).
      input_seg_seq: List of segmentation masks with seq_length elements of shape
                     (H, W, C) for some number of channels C.

    Returns:
      Input image stack with detections provided by segmentation mask removed.
    """
    background = [mask == 0 for mask in input_seg_seq]
    background = functools.reduce(lambda m1, m2: m1 & m2, background)
    # If masks are RGB, assume all channels to be the same. Reduce to the first.
    if background.ndim == 3 and background.shape[2] > 1:
        background = np.expand_dims(background[:, :, 0], axis=2)
    elif background.ndim == 2:  # Expand.
        background = np.expand_dism(background, axis=2)
    # background is now of shape (H, W, 1).
    background_stack = np.tile(background, [1, 1, input_image_stack.shape[3]])
    return np.multiply(input_image_stack, background_stack)


if __name__ == '__main__':
    # Converting it live
    # currently works on passing any image with its segmentation
    sess, inference_model = load_depth_model("../Trained-Models/depth-model/model-199160")
    img = cv2.imread('/home/bhushan/Videos/struct2depth/input/0.png')
    img_seg = cv2.imread('/home/bhushan/Videos/struct2depth/input/0-seg.png')
    image_list = [img, img_seg]

    object_motion = run_inference(sess=sess, model=inference_model, image_list=image_list)
