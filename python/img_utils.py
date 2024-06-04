import cv2
from PIL import Image
import numpy as np


def read_and_convert_PIL(image_uri):
    image_arr = np.asarray(Image.open(image_uri).convert('RGB'), dtype=np.uint8)
    return Image.fromarray(image_arr)


def read_and_convert_OCV_to_PIL(image_uri):
    img = cv2.imread(image_uri, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV default format is BGR
    return Image.fromarray(img_rgb)


def read_OCV_bytes(img_uri):
    img = cv2.imread(img_uri, cv2.IMREAD_COLOR)
    bpp = img.shape[2] if len(img.shape) > 2 else 1
    if bpp > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #      img_bytes,     width,        height,       bytes_per_pixel, bytes_per_line
    return img.tobytes(), img.shape[1], img.shape[0], bpp,             bpp*img.shape[1]


def preprocess(np_image):
    if len(np_image.shape) < 3:
        h, w = np_image.shape
        img = np_image  #.copy()
    else:
        h, w, c = np_image.shape
        img = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    return img


def detect_plates_ocv(image, cascade_uri):
    img = image.copy()  # or preprocess(image)
    carplate_haar_cascade = cv2.CascadeClassifier(cascade_uri)
    SF = 1.01
    MN = 4
    # TODO Parse cascade_uri to extract scaleFactor SF and minNeighbors MN
    plate_rects = carplate_haar_cascade.detectMultiScale(img, scaleFactor=SF, minNeighbors=MN)  # , minSize=(w, h)
    for x,y,w,h in plate_rects:  # 2D numpy array, each row is [x y w h]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    return plate_rects, img
