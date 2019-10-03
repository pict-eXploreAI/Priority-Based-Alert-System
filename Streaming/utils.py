from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
from imageio import imread

def pil_image_to_base64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())


def base64_to_pil_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))


def base64_to_cv2(base64_img):
	#nparr = np.fromstring(base64_img.decode('base64'), np.uint8)
	#return cv2.imdecode(nparr,cv2.IMREAD_COLOR)
	#img = imread(BytesIO(base64.b64decode(base64_img)))
	#return img
	pil_img = Image.open(BytesIO(base64.b64decode(base64_img)))
	return np.array(pil_img)

def cv2_to_base64(cv2Img):
	#buf = cv2.imencode(".jpeg",cv2Img)
	#return base64.b64encode(buf.getvalue())
	pil_image = Image.fromarray(cv2Img.astype('uint8'),'RGB')	
	buf = BytesIO()
	pil_image.save(buf, format="PNG")
	return base64.b64encode(buf.getvalue())
