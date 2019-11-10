import cv2
import numpy as np
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


def nothing(x):
    pass

# Create a black image, a window
# img = np.zeros((300,512,3), np.uint8)

def switches(switch):

    cv2.createTrackbar(switch, 'video', 0, 1, nothing)
    cv2.createTrackbar('H_MIN', 'video', H_MIN, H_MAX, nothing)
    cv2.createTrackbar('H_MAX', 'Video', H_MAX, H_MAX, nothing)
    cv2.createTrackbar('S_MIN', 'video', S_MIN, S_MAX, nothing)
    cv2.createTrackbar('S_MAX', 'Video', S_MAX, S_MAX, nothing)
    cv2.createTrackbar('V_MIN', 'video', V_MIN,  V_MAX, nothing)
    cv2.createTrackbar('V_MAX', 'Video', V_MAX, V_MAX, nothing)


# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'


# def apply_rescale(frame, percent):
#     frame_width = int(frame.shape[1] * percent/ 100)
#     frame_height = int(frame.shape[0] * percent/ 100)
#     dim = (frame_width, frame_height)
#     resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#     return resized_frame


def main ():
    camera = cv2.VideoCapture(0)

    cv2.namedWindow('video')

    frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH);
    frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT);

    switch = '0 : OFF \n1 : ON'
    switches(switch)

    while(1):
        ret, frame = camera.read()
        if not ret:
            break

        # get current positions of four trackbars
        hmin = cv2.getTrackbarPos('H_MIN','video')
        hmax = cv2.getTrackbarPos('H_MAX','video')
        vmin = cv2.getTrackbarPos('S_MIN','video')
        vmax = cv2.getTrackbarPos('S_MAX','video')
        smin = cv2.getTrackbarPos('S_MIN','video')
        smax = cv2.getTrackbarPos('S_MAX','video')

        s = cv2.getTrackbarPos(switch,'video')


        # switch to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # # find mask of pixels within HSV range
        # skinMask = cv2.inRange(hsv, lower, upper)
        #
        # # denoise
        # skinMask = cv2.GaussianBlur(skinMask, (9, 9), 0)
        #
        # # kernel for morphology operation
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        #
        # # CLOSE (dilate / erode)
        # skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel, iterations = 3)
        #
        # # denoise the mask
        # skinMask = cv2.GaussianBlur(skinMask, (9, 9), 0)
        #
        # # only display the masked pixels
        # skin = cv2.bitwise_and(frame, frame, mask = skinMask)
        # cv2.imshow("HSV", frame)

        # cv2.imshow('Video', frame)


        # resized_frame = apply_rescale(frame, 50) # downscale


        #
        # lower = np.array([hmin, vmin, smin], dtype="uint8")
        # upper = np.array([hmax,vmax,smax], dtype="uint8")
        #
        #
        # # find mask of pixels within HSV range
        # skinMask = cv2.inRange(hsv, lower, upper)
        #
        # cv2.imshow("HSV", hsv)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break




    camera.release()
    # out.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
   main()
