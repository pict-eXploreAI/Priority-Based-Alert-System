# Depth Model

This module finds the depth-map of given Image

## How to run only depth model
Run the command `python depth.py`, This will give the live depth-map output from the frames of depth map

## Directory Structure
depth - this is the actual inference file which generates the output<br/>
model - this is for loading the Model<br/>
nets - this is the actual architecture of the model (RESNET)<br/>
util - this file contains the common general purpose functions<br/>

## References
1. A pertained tensorflow model for scene depth prediction and depth map generation is
used. struct2depth - https://github.com/tensorflow/models/tree/master/research/struct2depth
2. The model is trained for unsupervised learning of scene depth where supervision is
provided only by monocular videos. Casser et al. 2018 - https://arxiv.org/pdf/1811.06152.pdf
