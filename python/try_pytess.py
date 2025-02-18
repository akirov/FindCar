import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from img_utils import detect_plates_ocv_haar, preprocess_plate, plt_imshow_actual_size

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


# Tesseract options:
#  --tessdata-dir PATH   Specify the location of tessdata path.
#  --user-words PATH     Specify the location of user words file.
#  --user-patterns PATH  Specify the location of user patterns file.
#  --dpi VALUE           Specify DPI for input image.
#  --loglevel LEVEL      Specify logging level. LEVEL can be
#                        ALL, TRACE, DEBUG, INFO, WARN, ERROR, FATAL or OFF.
#  -l LANG[+LANG]        Specify language(s) used for OCR.
#  -c VAR=VALUE          Set value for config variables.
#                        Multiple -c arguments are allowed.
#  --psm NUM             Specify page segmentation mode.
#  --oem NUM             Specify OCR Engine mode.
#  NOTE: These options must occur before any configfile.
# Languages:
#  '-l eng'  for using the English language
#  '-l bul'  for using the Bulgarian language
#  '-l rus'  for using the Russian language
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
# Tesseract config parameters.
#  tessedit_write_params_to_file Write all parameters to the given file.
#  tessedit_write_images     0       Capture the image from the IPE (see how Tesseract has processed the image; requires output parameter)
#  interactive_display_mode  0       Run interactively?
#  dawg_debug_level          0       Set to 1 for general debug info, to 2 for more details, to 3 to see all the debug messages
#  tessedit_create_pdf       0       Write .pdf output file
#  tessedit_ocr_engine_mode  3       Which OCR engine(s) to run (Tesseract, LSTM, both). Defaults to loading and running the most accurate available.
#  classify_bln_numeric_mode 0       Assume the input is numbers [0-9].
#  tessedit_init_config_only 0       Only initialize with the config file. Useful if the instance is not going to be used for OCR but say only for layout analysis.
#  user_words_file           A filename of user-provided words.
#  user_patterns_file        A filename of user-provided patterns.
#  user_patterns_suffix      A suffix of user-provided patterns located in tessdata.
#  tessedit_char_blacklist   Blacklist of chars not to recognize
#  tessedit_char_whitelist   Whitelist of chars to recognize
#  enable_noise_removal      1       Remove and conditionally reassign small outlines when they confuse layout analysis, determining diacritics vs noise
#  textord_heavy_nr          0       Vigorously remove noise
#  tessedit_font_id          0       Font ID to use or zero
#  tessedit_use_reject_spaces    1       Reject spaces?
#  tessedit_preserve_min_wd_len  2       Only preserve wds longer than this  ???
#  tessedit_reject_bad_qual_wds  1       Reject all bad quality wds
#  tessedit_unrej_any_wd     0       Don't bother with word plausibility
#  tessedit_flip_0O          1       Contextual 0O O0 flips
#  textord_noise_rejwords    1       Reject noise-like words
#  textord_noise_rejrows     1       Reject noise-like rows
#  tessedit_reject_row_percent   40      %rej allowed before rej whole row
#  tessedit_reject_block_percent 45      %rej allowed before rej whole block
#  tessedit_reject_doc_percent   65      %rej allowed before rej whole doc
#  max_permuter_attempts     10000   Maximum number of different character choices to consider during permutation. This limit is especially useful when user patterns are specified, since overly generic patterns can result in dawg search exploring an overly large number of options.
#  matcher_good_threshold    0.125   Good Match (0-1)  ???
#  load_system_dawg          1       Load system word dawg.
#  load_freq_dawg            1       Load frequent word dawg.
#  load_unambig_dawg         1       Load unambiguous word dawg.
#  load_punc_dawg            1       Load dawg with punctuation patterns.
#  load_number_dawg          1       Load dawg with number patterns.
#  load_bigram_dawg          1       Load dawg with special word bigrams.
#  language_model_penalty_non_freq_dict_word 0.1     Penalty for words not in the frequent word dictionary
#  language_model_penalty_non_dict_word      0.15    Penalty for non-dictionary words

my_config = '-l eng --psm 7'  #  --oem 3
my_config += ' -c tessedit_char_whitelist="ABCEHKMOPTXY0123456789 ."'
#my_config += ' -c user_patterns_suffix="patterns"'
#my_config += ' -c textord_noise_rejrows=0'


def detect_plates_tess(image, conf=''):
    img = image.copy()
    if not conf: conf = '--psm 11 -c tessedit_char_whitelist="ABCEHKMOPTXY0123456789 ."'
    image_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=conf)
    #print(image_data)
    plate_rects = []
    for i in range(0, len(image_data['conf'])):
        if image_data['conf'][i] < 0: continue  #  or not image_data['text'][i]
        x, y, w, h = image_data['left'][i], image_data['top'][i], image_data['width'][i], image_data['height'][i]
        plate_rects.append([x, y, w, h])
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    return plate_rects, img


def draw_boxes_txt(img, boxes_txt):
    h = img.shape[0]
    img_with_boxes = img.copy()
    for l in boxes_txt.splitlines():
        box = l.split(" ")
        character = box[0]
        x1 = int(box[1])
        y1 = int(box[2])
        x2 = int(box[3])
        y2 = int(box[4])
        cv2.rectangle(img_with_boxes, (x1, h - y1), (x2, h - y2), (0,255,0), 1)
        cv2.putText(img_with_boxes, character, (x1 + 2, h - y1 - 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    return img_with_boxes


def draw_data_dict(img, data_dict):
    # data_dict = {'level': [], 'page_num': [], 'block_num': [], 'par_num': [], 'line_num': [],\
    #              'word_num': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': [], 'text': []}
    img_with_boxes = img.copy()
    for i in range(len(data_dict['word_num'])):
        if data_dict['word_num'][i] == 0 or data_dict['conf'][i] < 0: continue
        (x, y, w, h) = (data_dict['left'][i], data_dict['top'][i], data_dict['width'][i], data_dict['height'][i])
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img_with_boxes, data_dict['text'][i], (x + 2, y + h - 2), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    return img_with_boxes


def read_plates(image, plate_rects, conf='', title='', preprocess=False, detect_tess_boxes=False):
    plate_txts = []
    #col = 200 if len(image.shape) < 3 else (0,255,0)
    for i, (x0,y0,w0,h0) in enumerate(plate_rects):
        plate = image[y0:y0+h0, x0:x0+w0]
        if title: plt.figure(num=title+f" [x={x0},y={y0},w={w0},h={h0}]")
        if preprocess:  # and h > ...
            plate = preprocess_plate(plate)
        if not conf: conf = my_config
        if detect_tess_boxes:
            text = ''
            image_data = pytesseract.image_to_data(plate, output_type=pytesseract.Output.DICT, config='--psm 11')
            for j in range(0, len(image_data['conf'])):
                if image_data['conf'][j] < 0: continue
                x, y, w, h = image_data['left'][j], image_data['top'][j], image_data['width'][j], image_data['height'][j]
                box = plate[y:y+h, x:x+w]
                txt = pytesseract.image_to_string(box, config=conf)
                text += (' ' if text and txt else '') + txt.strip()
                #cv2.rectangle(plate, (x,y), (x+w,y+h), col, 2)  # This modifies the plate!
        else:
            text = (pytesseract.image_to_string(plate, config=conf)).strip()
        plt.imshow(plate, cmap='gray' if len(plate.shape) < 3 else None)
        plt.show()
        print(f"{title} image[x={x0}:{x0+w0}, y={y0}:{y0+h0}], image_to_string('{conf}'): '{text}'")
        plate_txts.append(text)
    return plate_txts


def postprocess_plate_txt(plate_txt, country='BG'):
    # Remove starting or trailing numbers
    # Remove starting or trailing more than 2 letters
    # Replace '.' with ' '
    # X-Y processing
    # Match first letters from the possible choices (for the country)
    # Consider temporary and two-row numbers?
    return plate_txt


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} image_uri")
        exit(1)
    image_uri = sys.argv[1]


    # In order to bypass the image conversions of pytesseract, just use relative or absolute image path
    print(f"image_to_string({image_uri}) = '{pytesseract.image_to_string(image_uri)}'")
    print(f"image_to_string({image_uri}, config='{my_config}') = '{pytesseract.image_to_string(image_uri, config=my_config)}'")

    # Get information about orientation and script detection
    try:
        osd = pytesseract.image_to_osd(image_uri)
        print(f"image_to_osd({image_uri}):\n{osd}")
    except:
        print("Error in image_to_osd()")
    print()


    # Consider the whole image, using PIL
    image = Image.open(image_uri)
    #plt.figure(num='Whole image loaded with PIL')
    #plt.imshow(image)
    #plt.show()
    plt_imshow_actual_size(image, title='Whole image loaded with PIL')

    text_from_image = pytesseract.image_to_string(image, lang='eng')
    print(f"PIL image_to_string(lang='eng'): '{text_from_image}'")
    text_from_image = pytesseract.image_to_string(image, config=my_config)
    print(f"PIL image_to_string(config='{my_config}'): '{text_from_image}'")

    # Get bounding box estimates
    print(f"PIL image_to_boxes:\n{pytesseract.image_to_boxes(image)}")

    # Get verbose data including boxes, confidences, line and page numbers
    print(f"PIL image_to_data:\n{pytesseract.image_to_data(image)}")


    # Consider the whole image, using OpenCV
    img_bgr = cv2.imread(image_uri, cv2.IMREAD_COLOR)
    print(f"OpenCV imread '{image_uri}', (h,w,c)={img_bgr.shape}\n")
    #cv2.namedWindow('Whole image loaded with OpenCV', cv2.WINDOW_NORMAL)  # WINDOW_OPENGL
    #cv2.resizeWindow('Whole image loaded with OpenCV', 1+int(img_bgr.shape[1]*100/125), 1+int(img_bgr.shape[0]*100/125))
    cv2.imshow('Whole image loaded with OpenCV', img_bgr)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    #text = pytesseract.image_to_string(img_bgr, config=my_config)
    #print(f"OpenCV BGR image_to_string('{my_config}'): '{text}'\n")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # cv2.imread(image_uri, cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(img_gray, config=my_config)
    print(f"OpenCV Gray image_to_string('{my_config}'): '{text}'\n")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # OpenCV default format is BGR
    text = pytesseract.image_to_string(img_rgb, config=my_config)
    print(f"OpenCV RGB image_to_string('{my_config}'): '{text}'\n")

    boxes_txt = pytesseract.image_to_boxes(img_rgb, config=my_config)
    print(f"OpenCV RGB image_to_boxes:\n{boxes_txt}")
    img_rgb_with_char_boxes = draw_boxes_txt(img_rgb, boxes_txt)
    plt_imshow_actual_size(img_rgb_with_char_boxes, title='OpenCV img_rgb_with_char_boxes')

    data_txt = pytesseract.image_to_data(img_rgb, config=my_config)
    print(f"OpenCV RGB image_to_data:\n{data_txt}")
    data_dict = pytesseract.image_to_data(img_rgb, config=my_config, output_type=pytesseract.Output.DICT)
    img_rgb_with_word_boxes = draw_data_dict(img_rgb, data_dict)
    plt_imshow_actual_size(img_rgb_with_word_boxes, title='OpenCV img_rgb_with_word_boxes')


    # Plate recognition pipeline:
    # 1. Detect number plates using Tesseract image_to_data() or better with Haar cascade classifier
    # 2. Pre-process plate boxes (optional): filter by size, find intersecting and one-into-another
    #  boxes, then for sub-boxes keep inner ones (with smaller area), for intersecting boxes take
    #  the wrapping box, auto-crop, filter by color, call pytesseract.image_to_data() inside Haar boxes
    # 3. Pre-process plates: convert to grayscale (needed for Otsu binarization), invert (if needed) to
    #  have black text on white, filter noise, binarize, erode and/or dilate, crop more, rotate, scale...
    # 4. Recognize text with Tesseract
    # 5. Post-process the result: remove starting or trailing numbers, remove starting or trailing more
    #  than 2 letters, replace '.' with ' ', X-Y processing, country-specific processing

    # Detect plates' positions using Tesseract and read them
    plate_rects_ts, img_with_plates_ts = detect_plates_tess(img_rgb)
    print("plate_rects_ts [[x,y,w,h]]:\n", plate_rects_ts)
    plt_imshow_actual_size(img_with_plates_ts, title='OpenCV RGB image with detect_plates_tess boxes')
    read_plates(img_rgb, plate_rects_ts, conf=my_config, title='detect_plates_tess')
    print()

    # Detect plates' positions with OpenCV Haar cascade and read them
    plate_rects_cv, img_with_plates_cv = detect_plates_ocv_haar(img_rgb,
        '../opencv/data/haarcascades/haarcascade_russian_plate_number_[SF=1.01]_[MN=4].xml',
        auto_crop=True)  #, img_uri=image_uri
    print("plate_rects_cv [[x,y,w,h]]:\n", plate_rects_cv)
    plt_imshow_actual_size(img_with_plates_cv, title='OpenCV RGB image with detect_plates_ocv_haar boxes')
    read_plates(img_rgb, plate_rects_cv, conf=my_config, title='detect_plates_ocv_haar', preprocess=False)  #, detect_tess_boxes=True
    plate_txts = read_plates(img_rgb, plate_rects_cv, conf=my_config, title='detect_plates_ocv_haar_processed', preprocess=True)
    for txt in plate_txts:
        plate_txt = postprocess_plate_txt(txt)
    print()

    plate_rects_cv_gray, img_gray_with_plates_cv = detect_plates_ocv_haar(img_gray,
        '../opencv/data/haarcascades/haarcascade_russian_plate_number_[SF=1.01]_[MN=4].xml',
        auto_crop=True)  #, img_uri=image_uri
    print("plate_rects_cv_gray [[x,y,w,h]]:\n", plate_rects_cv_gray)
    plt_imshow_actual_size(img_gray_with_plates_cv, title='OpenCV Gray image with detect_plates_ocv_haar boxes')
    read_plates(img_gray, plate_rects_cv_gray, conf=my_config, title='detect_plates_ocv_haar_gray', preprocess=False)
    read_plates(img_gray, plate_rects_cv_gray, conf=my_config, title='detect_plates_ocv_haar_gray_processed', preprocess=True)
