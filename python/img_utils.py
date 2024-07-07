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


def save_bool_nparray_as_binary_png(bool_nparray, save_uri):
    pil_img = Image.fromarray(bool_nparray)
    pil_img.save(save_uri, bits=1, optimize=True)


def detect_plates_ocv_haar(image, cascade_uri):
    img = image.copy()
    carplate_haar_cascade = cv2.CascadeClassifier(cascade_uri)
    SF = 1.01
    MN = 4
    # TODO Parse cascade_uri to extract scaleFactor SF and minNeighbors MN
    plate_rects = carplate_haar_cascade.detectMultiScale(img, scaleFactor=SF, minNeighbors=MN)  # , minSize=(w, h)
    col = 255 if len(img.shape) < 3 else (255,0,0)
    for x,y,w,h in plate_rects:  # 2D numpy array, each row is [x y w h]
        cv2.rectangle(img, (x,y), (x+w,y+h), col, 2)
    # TODO Detect overlapping and one into another Haar regions and filter them
    return plate_rects, img


def preprocess_plate(np_image):
    if len(np_image.shape) < 3:
        img_gray = np_image  #.copy()
    else:
        img_gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        #img_HSV = cv2.cvtColor(np_image, cv.COLOR_RGB2HSV)
        #img_bw = cv2.inRange(np_image, (128, 128, 128), (255, 255, 255))
    thresh = 127
    img_bw = img_gray > thresh
    #(thresh, img_bw) = cv2.threshold(np_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_bw
