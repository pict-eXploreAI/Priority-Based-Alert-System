# Priority evaluation

This module finds the priority wise message list of the given depth-map and segmentation mask

## How to run only Priority
1. Store the depth-maps in depth folder and segmentation masks in the segment folder
2. Run the command `python priority.py`
3. Results will be stored in the output folder

## Logic for finding the priority based message list
- First we will get the image of depth-map and corresponding segmentation mask
- We will then resize both the images to (416, 128, 1) shape
- Then the segmentation mask image is converted to the weighted image based on priorities based on human intuition
- The weighted image is multiplied with the depth-map to get the priority values of each pixel
- This final image is looped over to get the final message list
