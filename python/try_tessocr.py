import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from img_utils import read_and_convert_PIL, read_and_convert_OCV_to_PIL

if 'TESSDATA_PREFIX' not in os.environ:
    os.environ['TESSDATA_PREFIX'] = r'../tesseract/tessdata'

import tesserocr as tocr

print(tocr.get_languages())


images = ['../data/test/bul_001.jpg', '../data/test/bul_001_seg.jpg']  # , '../data/test/bul_001_seg.png'

for img_uri in images:
    print("\nfile_to_text(", img_uri, ") = '", tocr.file_to_text(img_uri), "'")
    image = Image.open(img_uri)
    plt.imshow(image)
    plt.show()
    print("PIL image_to_text(", img_uri, ") = '", tocr.image_to_text(image), "'")

print("\n\nUsing PyTessBaseAPI")
with tocr.PyTessBaseAPI(lang='eng', psm=7, oem=3) as api:
    api.SetVariable("tessedit_char_whitelist", "ABCEHKMOPTXY0123456789 .")
    for img_uri in images:
        api.SetImageFile(img_uri)
        print("\nSetImageFile: ", img_uri)
        print("GetUTF8Text: '", api.GetUTF8Text(), "'")
        print("Confidence per word: ", api.AllWordConfidences())

        api.SetImage(Image.open(img_uri))
        print("SetImage(PIL image): ", img_uri)
        print("GetUTF8Text: '", api.GetUTF8Text(), "'")  # Not recognized without psm=7
        print("Confidence per word: ", api.AllWordConfidences())

        api.SetImage(read_and_convert_PIL(img_uri))
        print("SetImage(read_and_convert_PIL): ", img_uri)
        print("GetUTF8Text: '", api.GetUTF8Text(), "'")
        print("Confidence per word: ", api.AllWordConfidences())

        api.SetImage(read_and_convert_OCV_to_PIL(img_uri))
        print("SetImage(read_and_convert_OCV_to_PIL): ", img_uri)
        print("GetUTF8Text: '", api.GetUTF8Text(), "'")
        print("Confidence per word: ", api.AllWordConfidences())

# api is automatically finalized when used in a with-statement (context manager).
# otherwise api.End() should be explicitly called when it's no longer needed.
