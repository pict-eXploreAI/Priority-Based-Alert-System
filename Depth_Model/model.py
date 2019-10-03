""" Build model for the inferences """

import sys
from absl import logging
import numpy as np
import tensorflow as tf
sys.path.append("../")
from Depth_Model.nets import RESNET, disp_net, objectmotion_net

gfile = tf.gfile
slim = tf.contrib.slim

# See nets.encoder_resnet as reference for below input-normalizing constants.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_SD = (0.229, 0.224, 0.225)


class Model(object):
  """ Model code based on SfMLearner """

  def __init__(self,
               file_extension='png',
               batch_size=1, 
               img_height=128,
               img_width=416,
               seq_length=3,
               architecture=RESNET,
               imagenet_norm=True,
               weight_reg=0.05,
               use_skip=True,
               joint_encoder=True):
    # load the variables to the object
    self.file_extension = file_extension
    self.batch_size = batch_size
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.architecture = architecture
    self.imagenet_norm = imagenet_norm
    self.weight_reg = weight_reg
    self.use_skip = use_skip
    self.joint_encoder = joint_encoder

    self.build_depth_test_graph()
    self.build_objectmotion_test_graph()

  def build_depth_test_graph(self):
    """ Builds depth model reading from placeholders """
    with tf.variable_scope('depth_prediction'):
      input_image = tf.placeholder(
          tf.float32, [self.batch_size, self.img_height, self.img_width, 3],
          name='raw_input')
      if self.imagenet_norm:
        input_image = (input_image - IMAGENET_MEAN) / IMAGENET_SD
      est_disp, _ = disp_net(architecture=self.architecture,
                                  image=input_image,
                                  use_skip=self.use_skip,
                                  weight_reg=self.weight_reg,
                                  is_training=True)
    est_depth = 1.0 / est_disp[0]
    self.input_image = input_image
    self.est_depth = est_depth

  def build_objectmotion_test_graph(self):
      """Builds egomotion model reading from placeholders."""
      input_image_stack_om = tf.placeholder(
          tf.float32,
          [1, self.img_height, self.img_width, self.seq_length * 3],
          name='raw_input')

      if self.imagenet_norm:
        im_mean = tf.tile(
            tf.constant(IMAGENET_MEAN), multiples=[self.seq_length])
        im_sd = tf.tile(
            tf.constant(IMAGENET_SD), multiples=[self.seq_length])
        input_image_stack_om = (input_image_stack_om - im_mean) / im_sd

      with tf.variable_scope('objectmotion_prediction'):
        est_objectmotion = objectmotion_net(
            image_stack=input_image_stack_om,
            disp_bottleneck_stack=None,
            joint_encoder=False,
            seq_length=self.seq_length,
            weight_reg=self.weight_reg)
      self.input_image_stack_om = input_image_stack_om
      self.est_objectmotion = est_objectmotion

  def inference_depth(self, inputs, sess):
    '''
    Input: 1. inputs - get the image on which depth to be inferred
           2. sess - session in which the model variables are loaded
    Output: returns the depth-map of inputs
    '''
    return sess.run(self.est_depth, feed_dict={self.input_image: inputs})

  def inference_objectmotion(self, inputs, sess):
    return sess.run(
        self.est_objectmotion, feed_dict={self.input_image_stack_om: inputs})