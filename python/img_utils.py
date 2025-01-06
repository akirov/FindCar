import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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


def plt_imshow_actual_size(image, dpi=125, title=None):
    # What size does the figure need to be in inches to fit the image?
    if isinstance(image, Image.Image):
        figsize = image.width / float(dpi), image.height / float(dpi)
        cmap = 'gray' if len(image.getbands()) == 1 else None
    else:  # numpy array
        figsize = image.shape[1] / float(dpi), image.shape[0] / float(dpi)
        cmap = 'gray' if len(image.shape) == 2 else None
        # Also convert BGR to RGB?

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize, num=title)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    ax.imshow(image, cmap=cmap)
    plt.show()


def split_uri(uri):
    name_pos = 0
    for i in range(len(uri)-1, -1, -1):
        c = uri[i]
        if c == '/' or c == '\\':
            name_pos = i+1
            break
    file_path = uri[:name_pos]
    filename_ext = uri[name_pos:]
    dot_pos = filename_ext.rfind('.')
    filename = filename_ext[:dot_pos] if dot_pos > 0 else filename_ext
    return file_path, filename_ext, filename


def detect_plates_ocv_haar(image, cascade_uri, img_uri='', min_size=(60,20), max_size=(65535,21845),
                           auto_crop=False):
    img = image.copy()

    carplate_haar_cascade = cv2.CascadeClassifier(cascade_uri)
    SF = 1.03  # old: 1.01, default: 1.1
    MN = 2  # old: 4, default: 3
    # TODO Parse cascade_uri to extract scaleFactor SF and minNeighbors MN
    plate_rects = carplate_haar_cascade.detectMultiScale(img, scaleFactor=SF, minNeighbors=MN,
                                                         minSize=min_size, maxSize=max_size)

    img_base_uri = ''
    if img_uri:
        imgpath, _, imgname = split_uri(img_uri)
        img_base_uri = imgpath + imgname

    # Draw and save (optional) plate boxes
    cmap='gray' if len(image.shape) < 3 else None
    col1 = 255 if len(img.shape) < 3 else (255,0,0)
    for x,y,w,h in plate_rects:  # 2D numpy array, each row is [x y w h]
        cv2.rectangle(img, (x,y), (x+w,y+h), col1, 2)
        if img_base_uri and not auto_crop:
            box = image[y:y+h, x:x+w]
            box_uri = f"{img_base_uri}_{x},{y},{w},{h}.png"
            plt.imsave(box_uri, box, cmap=cmap)

    # Find intersecting and one-into-another boxes, then:
    # - for sub-boxes keep inner ones (with smaller area, but not too small), remove wrapping box?
    # - for intersecting boxes create wrapping box instead of them (unite them)? No.

    # Auto-crop Haar boxes with proper coefficients, because boxes are bigger than
    # the plate with certain proportion depending on size
    if auto_crop:
        col2 = 200 if len(img.shape) < 3 else (200,0,0)
        cx = 0.03  # varies a lot!
        cy = 0.25
        cw = 0.97  # varies
        ch = 0.56
        for i, (x0,y0,w0,h0) in enumerate(plate_rects):
            #if h0 < 25: continue
            x = x0 + int(cx*w0)
            y = y0 + int(cy*h0)
            w = int(cw*w0)
            h = int(ch*h0)
            plate_rects[i] = (x,y,w,h)
            cv2.rectangle(img, (x,y), (x+w,y+h), col2, 2)
            if img_base_uri:
                box = image[y:y+h, x:x+w]
                box_uri = f"{img_base_uri}_{x},{y},{w},{h}.png"
                plt.imsave(box_uri, box, cmap=cmap)

    # Filter by color histogram?

    return plate_rects, img


def preprocess_plate(np_image, convertion=cv2.COLOR_RGB2GRAY, scale=False):
    # To facilitate Tesseract OCR, pre-process number plate regions:
    # - scale down large images, because recognition doesn't work well for letters above 70pt, optimal: 20-40 pt
    # - convert to gray (needed for Otsu) and invert (if needed), to have black text on white background
    # - filter noise (dust), if it can be done fast: GaussianBlur, medianBlur, ...
    # - convert to binary: simple thresholding (only for ideal images), adaptive (for small letters),
    #  Otsu (for bigger letters)
    # - erode (for white on black) or dilate (for black on white) to remove small particles
    # - detect plate borders and crop everything outside them (especially trailing noise) by:
    #   -- auto-crop based on statistics, or call pytesseract.image_to_data() in advance
    #   -- find the biggest connected white area and crop everything outside it (outside-in)
    #   -- start from the white at the center and expand it to plate boundaries (inside-out)
    #   -- detect plate rectangle (straight lines)? this may not be possible...
    #   -- crop by color elements, like the blue country area or flag colors?
    # - try different rotations to level the text (detect straight lines angle first)?

    max_h = 85  # ~70p letters
    opt_h = 25  # ~20p letters
    if np_image.shape[0] > max_h or (scale and np_image.shape[0] > opt_h):
        factor = opt_h / np_image.shape[0]
        np_image = cv2.resize(np_image, (0,0), fx=factor, fy=factor)  #, interpolation = cv2.INTER_NEAREST

    if convertion in [cv2.COLOR_RGB2HSV, cv2.COLOR_BGR2HSV]:
        if len(np_image.shape) == 2:
            img_bgr = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
            img_HSV = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            # Or just exit with an error?
        else:
            img_HSV = cv2.cvtColor(np_image, convertion)
        #_, _, img_gray = cv2.split(img_HSV)
        img_bw = cv2.inRange(img_HSV, (128, 128, 128), (255, 255, 255))
    else:
        if len(np_image.shape) == 2:
            img_gray = np_image  #.copy()
        else:
            img_gray = cv2.cvtColor(np_image, convertion)

        #thresh = 127
        #(thresh, img_bw) = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)  #img_bw = img_gray > thresh

        #block_size = 11  # Make it proportional to image size?
        #adjust_avg = -5  # Adapt to brightness?
        #img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, \
        #                               block_size, adjust_avg)  # cv2.ADAPTIVE_THRESH_GAUSSIAN_C

        #img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
        (thresh, img_bw) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  #img_blur
        #print("preprocess_plate() threshold: {}".format(thresh))

    return img_bw
