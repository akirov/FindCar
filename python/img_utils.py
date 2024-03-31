import cv2

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
