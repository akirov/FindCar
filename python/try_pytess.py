import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import cv2

TESSERACT_PREFIX = r'../tesseract'
#if 'TESSDATA_PREFIX' not in os.environ:
os.environ['TESSDATA_PREFIX'] = TESSERACT_PREFIX + r'/tessdata'

import pytesseract

# Set the location of tesseract exe file
if sys.platform in ['win32', 'msys']:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PREFIX + r'/bin/mingw_win32/bin/tesseract.exe'
else:
    print("Please specify the location of tesseract exe file")
    exit(1)


def detect_plates_ocv(image, cascade_uri):
    carplate_haar_cascade = cv2.CascadeClassifier(cascade_uri)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    SF = 1.01
    MN = 4
    # TODO Parse cascade_uri to extract scaleFactor SF and minNeighbors MN
    plate_rects = carplate_haar_cascade.detectMultiScale(img, scaleFactor=SF, minNeighbors=MN)  # , minSize=(w, h)
    for x,y,w,h in plate_rects:  # 2D numpy array, each row is [x y w h]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    return plate_rects, img


def detect_plates_tess(image, conf=''):
    img = image.copy()
    if not conf: conf = '--psm 11'
    image_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=conf)
    #print(image_data)
    plate_rects = []
    for i in range(0, len(image_data['conf'])):
        if image_data['conf'][i] < 0: continue  #  or not image_data['text'][i]
        x, y, w, h = image_data['left'][i], image_data['top'][i], image_data['width'][i], image_data['height'][i]
        plate_rects.append([x, y, w, h])
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    return plate_rects, img


def read_plates(image, plate_rects, conf='', title=''):
    for x,y,w,h in plate_rects:
        plate = image[y:y+h, x:x+w]
        if title: plt.figure(num=title+f" [x={x},y={y},w={w},h={h}]")
        plt.imshow(plate)
        plt.show()
        if not conf: conf = '-l eng --oem 3 --psm 7'
        text = pytesseract.image_to_string(plate, config=conf)
        print(f"image['{y}':'{y+h}', '{x}':'{x+w}'] image_to_string('{config}'): '{text}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image_uri")
        exit(1)
    image_uri = sys.argv[1]


    # In order to bypass the image conversions of pytesseract, just use relative or absolute image path
    print(f"image_to_string({image_uri}) = '{pytesseract.image_to_string(image_uri)}'")

    # Get information about orientation and script detection
    try:
        osd = pytesseract.image_to_osd(image_uri)
        print(f"image_to_osd({image_uri}):\n{osd}")
    except:
        print("Error in image_to_osd()")


    # Consider the whole image, using PIL
    image = Image.open(image_uri)
    plt.figure(num='Whole image loaded with PIL')
    plt.imshow(image)
    plt.show()

    text_from_image = pytesseract.image_to_string(image, lang='eng')
    print(f"PIL image_to_string(lang='eng'): '{text_from_image}'")

    # Get bounding box estimates
    print(f"PIL image_to_boxes:\n{pytesseract.image_to_boxes(image)}")

    # Get verbose data including boxes, confidences, line and page numbers
    print(f"PIL image_to_data:\n{pytesseract.image_to_data(image)}")


    # Tesseract config parameters.
    # '-l eng'  for using the English language
    # OCR Engine modes:
    #  0    Legacy engine only.
    #  1    Neural nets LSTM engine only.
    #  2    Legacy + LSTM engines.
    #  3    Default, based on what is available.
    # Page segmentation modes:
    #  0    Orientation and script detection (OSD) only.
    #  1    Automatic page segmentation with OSD.
    #  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
    #  3    Fully automatic page segmentation, but no OSD. (Default)
    #  4    Assume a single column of text of variable sizes.
    #  5    Assume a single uniform block of vertically aligned text.
    #  6    Assume a single uniform block of text.
    #  7    Treat the image as a single text line.
    #  8    Treat the image as a single word.
    #  9    Treat the image as a single word in a circle.
    # 10    Treat the image as a single character.
    # 11    Sparse text. Find as much text as possible in no particular order.
    # 12    Sparse text with OSD.
    # 13    Raw line. Treat the image as a single text line,
    #       bypassing hacks that are Tesseract-specific.
    config = '-l eng --oem 3 --psm 7'


    # Consider the whole image, using OpenCV
    img = cv2.imread(image_uri, cv2.IMREAD_COLOR)
    cv2.imshow('Whole image loaded with OpenCV', img)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    text = pytesseract.image_to_string(img, config=config)
    print(f"OpenCV BGR image_to_string('{config}'): '{text}'")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV default format is BGR
    text = pytesseract.image_to_string(img_rgb, config=config)
    print(f"OpenCV RGB image_to_string('{config}'): '{text}'")
    print(f"OpenCV RGB image_to_boxes:\n{pytesseract.image_to_boxes(img_rgb, config=config)}")
    print(f"OpenCV RGB image_to_data:\n{pytesseract.image_to_data(img_rgb, config=config)}")


    # Detect text position
    plate_rects_cv, img_with_plates_cv = detect_plates_ocv(img, '../opencv/data/haarcascades/haarcascade_russian_plate_number_[SF=1.01]_[MN=4].xml')
    print(plate_rects_cv)
    cv2.imshow('OpenCV GRAY image with detect_plates_ocv boxes', img_with_plates_cv)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    read_plates(img_rgb, plate_rects_cv, conf=config, title='detect_plates_ocv')

    plate_rects_ts, img_with_plates_ts = detect_plates_tess(img_rgb)
    print(plate_rects_ts)
    plt.figure(num='OpenCV RGB image with detect_plates_tess boxes')
    plt.imshow(img_with_plates_ts)
    plt.show()
    read_plates(img_rgb, plate_rects_ts, conf=config, title='detect_plates_tess')
