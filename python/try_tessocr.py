import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import cv2

if 'TESSDATA_PREFIX' not in os.environ:
    os.environ['TESSDATA_PREFIX'] = r'../tesseract/tessdata'

import tesserocr as tocr


print(tocr.get_languages())

images = ['../data/test/bul_001.jpg', '../data/test/bul_001_seg.jpg', '../data/test/bul_001_seg.png']

with tocr.PyTessBaseAPI() as api:
    for img in images:
        api.SetImageFile(img)
        print("Image: ", img)
        print("Text: '", api.GetUTF8Text(), "'")
        print("Confidence per word: ", api.AllWordConfidences())
# api is automatically finalized when used in a with-statement (context manager).
# otherwise api.End() should be explicitly called when it's no longer needed.

for img in images:
    print(img, tocr.file_to_text(img))
    image = Image.open(img)
    plt.imshow(image)
    plt.show()
    print(img, tocr.image_to_text(image))
