""" Contains common utilities and functions """

import locale
import os
import re
from absl import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
gfile = tf.gfile

CMAP_DEFAULT = 'plasma'


def gray2rgb(im, cmap=CMAP_DEFAULT):
  '''
  Input: 1. im - input gray scale image
         2. cmap - the default cmap to convert im into colored image
  Output: returns the colored RGB image of gray scale
  '''
  cmap = plt.get_cmap(cmap)
  result_img = cmap(im.astype(np.float32))
  if result_img.shape[2] > 3:
    result_img = np.delete(result_img, 3, 2)
  return result_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None,
                                cmap=CMAP_DEFAULT):
  '''
  Input: 1. depth - depth-matrix of the image
  Output: returns the disparity matrix
  Description:
    Converts the depth-matrix into disparity-matrix by taking inverse
  '''
  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  disp = np.clip(disp, 0, 1)
  disp = gray2rgb(disp, cmap=cmap)
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[:keep_h]
  return disp


def get_vars_to_save_and_restore(ckpt=None):
  '''
  Input:  1. ckpt - Path to existing checkpoint.  If present, returns only the subset of
                    variables that exist in given checkpoint.
  Output: Returns list of variables that should be saved/restored
  Description:
    List of all variables that need to be saved/restored.
  '''
  model_vars = tf.trainable_variables()
  # Add batchnorm variables.
  bn_vars = [v for v in tf.global_variables()
             if 'moving_mean' in v.op.name or 'moving_variance' in v.op.name or
             'mu' in v.op.name or 'sigma' in v.op.name or
             'global_scale_var' in v.op.name]
  model_vars.extend(bn_vars)
  model_vars = sorted(model_vars, key=lambda x: x.op.name)
  mapping = {}
  if ckpt is not None:
    ckpt_var = tf.contrib.framework.list_variables(ckpt)
    ckpt_var_names = [name for (name, unused_shape) in ckpt_var]
    ckpt_var_shapes = [shape for (unused_name, shape) in ckpt_var]
    not_loaded = list(ckpt_var_names)
    for v in model_vars:
      if v.op.name not in ckpt_var_names:
        # For backward compatibility, try additional matching.
        v_additional_name = v.op.name.replace('egomotion_prediction/', '')
        if v_additional_name in ckpt_var_names:
          # Check if shapes match.
          ind = ckpt_var_names.index(v_additional_name)
          if ckpt_var_shapes[ind] == v.get_shape():
            mapping[v_additional_name] = v
            not_loaded.remove(v_additional_name)
            continue
          else:
            logging.warn('Shape mismatch, will not restore %s.', v.op.name)
        logging.warn('Did not find var %s in checkpoint: %s', v.op.name,
                     os.path.basename(ckpt))
      else:
        # Check if shapes match.
        ind = ckpt_var_names.index(v.op.name)
        if ckpt_var_shapes[ind] == v.get_shape():
          mapping[v.op.name] = v
          not_loaded.remove(v.op.name)
        else:
          logging.warn('Shape mismatch, will not restore %s.', v.op.name)
    if not_loaded:
      logging.warn('The following variables in the checkpoint were not loaded:')
      for varname_not_loaded in not_loaded:
        logging.info('%s', varname_not_loaded)
  else:  # just get model vars.
    for v in model_vars:
      mapping[v.op.name] = v
  return mapping
