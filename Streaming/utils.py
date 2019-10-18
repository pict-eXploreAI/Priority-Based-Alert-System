from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
from imageio import imread

# Encode PIL image to Base64 encoding
def pil_image_to_base64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())


# Decode Base64 image to PIL image
def base64_to_pil_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))


# Encode CV2 image to Base64 image
def cv2_to_base64(cv2Img):
	pil_image = Image.fromarray(cv2Img.astype('uint8'),'RGB')	
	buf = BytesIO()
	pil_image.save(buf, format="PNG")
	return base64.b64encode(buf.getvalue())
	

# Decode Base64 image to CV2 image
def base64_to_cv2(base64_img):
	pil_img = Image.open(BytesIO(base64.b64decode(base64_img)))
	return np.array(pil_img)
