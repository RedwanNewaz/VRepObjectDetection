import cv2
from ObjectDetector import  ObjectDetector
if __name__ == '__main__':
    # read image from hard drive
    img = cv2.imread('img.png')
    # initialize overlay image class for display bounding boxes with labels
    labels = '../ObjectDetector/mscoco_label_map.pbtxt'

    # initialize detector
    # check other supported object detectors  from ObjectDetector > __ini__.py
    detector = ObjectDetector('SSDmobileNet', labels)
    img = detector(img)
    # see result
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Display', (1280, 720))
    cv2.imshow('Display', img)
    cv2.waitKey(0)


