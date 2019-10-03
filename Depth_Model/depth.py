""" This python program runs the struct2depth model to find the depth-map of given image """

import sys
sys.path.append("../")
from Depth_Model.model import Model
import numpy as np
import tensorflow as tf
from Depth_Model.util import normalize_depth_for_display, get_vars_to_save_and_restore
import cv2
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # creating session
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # loads the model into session
    saver.restore(sess, model_ckpt)
    return sess,inference_model


# Function gives the depth-map output of the given frame
def run_inference(model=None,
                   frame=None,
                   sess=None,
                   img_width=416,
                   img_height=128):
    '''
    Input: 1. model - the pretrained-model of struct2depth
           2. frame - the image for which depth-map to be find out
           3. sess - the session in which the weights of the models are loaded
           4. img_width - the width of the output image
           5. img_height - the height of output image
           This model is trained for (416, 128) shape images
    Output: color_map - this is the depth-map of the given frame
    Description:
        First the given image is reshaped into (416, 128) shape using INTER_AREA interpolation, 
        we pass this image to the inference_depth function which gives the depth-map matrix of objects.
        This matrix is then normalized into 0-1 using normalize_depth_for_display method and then it is
        stored in the color_map which is the 3-channel color-map of given frame
    '''
    # Resizing the given frame in (416, 128) using interpolation
    final_image = cv2.resize(
        frame, (img_width, img_height), interpolation=cv2.INTER_AREA)

    # Estimating depth for reshaped frame
    est_depth = model.inference_depth([final_image], sess)

    # Converting the depth matrix into color-map
    color_map = normalize_depth_for_display(
        np.squeeze(est_depth[0]))
    return color_map



if __name__=="__main__":
    sess, inference_model = load_depth_model("../Trained-Models/depth-model/model-199160")

    # start the video capture of the device
    cap = cv2.VideoCapture(0)
    # find the depth frame by frame
    import time
    while True:
        start = time.time()
        ret_val, frame = cap.read()
        # get the color-map for each depth
        color_map = run_inference(sess=sess, model=inference_model, frame=frame)
        cv2.imshow("Depth", color_map)
        print("Inference time:", time.time() - start)
        # run the loop till ESC is pressed
        if cv2.waitKey(1) == 27:
            break
    # destroys all the windows created
    cv2.destroyAllWindows()
    # close the session
    sess.close()
