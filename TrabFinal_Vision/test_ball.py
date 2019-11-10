from tkinter import X
from tkinter.ttk import Style
#import PIL.Image, PIL.ImageTk
from PIL import Image, ImageEnhance
import tkinter as tk
import numpy as np
import cv2
import math

H_MIN = 0;
H_MAX = 256;
S_MIN = 0;
S_MAX = 256;
V_MIN = 0;
V_MAX = 256;
# default capture width and height
FRAME_WIDTH = 640;
FRAME_HEIGHT = 480;
# max number of objects to be detected in frame
MAX_NUM_OBJECTS=50;
# minimum and maximum object area
MIN_OBJECT_AREA = 20*20;
MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;



def switches(switch):
    cv2.createTrackbar(switch, 'Video Frame', 0, 1, applyVal)
    cv2.createTrackbar('lower', 'Video Frame', 0, 255, applyVal)
    cv2.createTrackbar('upper', 'Video Frame', 0, 255, applyVal)
    cv2.createTrackbar('Blur', 'Video Frame', 3, 5, applyVal)


#
# # determine upper and lower HSV limits for (my) skin tones



def main():
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Original Output')

    switch = '0 : OFF \n1 : ON'
    switches(switch)

    while (True):
        ret, frame = camera.read()
        if not ret:
            continue

        lower_i = cv2.getTrackbarPos('lower', 'Original Output')
        upper_i = cv2.getTrackbarPos('upper', 'Original Output')
        s = cv2.getTrackbarPos(switch, 'Original Output')

        lower = np.array([0, 100, 0], dtype="uint8")
        upper = np.array([50,255,255], dtype="uint8")
        switches(switch)


        # switch to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
          # find mask of pixels within HSV range
        skinMask = cv2.inRange(hsv, lower, upper)


        cv2.imshow("Original", frame)
        cv2.imshow("HSV", hsv)
        cv2.imshow("skinMask", skinMask)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def applyVal(value):
    print('Applying blur!', value)


if __name__ == '__main__':
   main()
