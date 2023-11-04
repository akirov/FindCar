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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image_uri")
        exit(1)
    image_uri = sys.argv[1]


    # In order to bypass the image conversions of pytesseract, just use relative or absolute image path
    print(f"image_to_string({image_uri}) = '{pytesseract.image_to_string(image_uri)}'")


    # Using PIL
    image = Image.open(image_uri)
    plt.imshow(image)
    plt.show()

    text_from_image = pytesseract.image_to_string(image, lang='eng')
    print(f"PIL result (default): '{text_from_image}'")

    # Get bounding box estimates
    print("image_to_boxes: ", pytesseract.image_to_boxes(image))

    # Get verbose data including boxes, confidences, line and page numbers
    print("image_to_data: ", pytesseract.image_to_data(image))

    # Get information about orientation and script detection
    #print("image_to_osd: ", pytesseract.image_to_osd(image))  # Error: "Too few characters"


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
    #   bypassing hacks that are Tesseract-specific.
    config = '-l eng --oem 3 --psm 7'


    # Using OpenCV
    im = cv2.imread(image_uri, cv2.IMREAD_COLOR)
    cv2.imshow('text image', im)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

    text = pytesseract.image_to_string(im, config=config)
    print(f"OpenCV result ('-l eng --oem 3 --psm 7'): '{text}'")
