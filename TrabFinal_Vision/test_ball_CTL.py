from tkinter import X
from tkinter.ttk import Style
#import PIL.Image, PIL.ImageTk
import imutils as imutils
from PIL import Image, ImageEnhance
import tkinter as tk
# import math
# from collections import deque
# from imutils.video import VideoStream
import numpy as np
# import argparse
import cv2


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


#
# def switches(switch):
#     cv2.createTrackbar(switch, 'Video Frame', 0, 1, applyVal)
#     cv2.createTrackbar('lower', 'Video Frame', 0, 255, applyVal)
#     cv2.createTrackbar('upper', 'Video Frame', 0, 255, applyVal)
#     cv2.createTrackbar('Blur', 'Video Frame', 3, 5, applyVal)


#
# # determine upper and lower HSV limits for (my) skin tones



def main():
    camera = cv2.VideoCapture("video.mp4")
    # camera = cv2.VideoCapture(0)
    cv2.namedWindow('Original Output')
    #
    # switch = '0 : OFF \n1 : ON'
    # switches(switch)

    while (True):
        ret, frame = camera.read()
        if not ret:
            continue
        #
        # lower_i = cv2.getTrackbarPos('lower', 'Original Output')
        # upper_i = cv2.getTrackbarPos('upper', 'Original Output')
        # s = cv2.getTrackbarPos(switch, 'Original Output')

        # referencias de cores para a bola verde
        # low_green = np.array([35, 40, 19])
        # up_green = np.array([82, 246, 139])

        lower = np.array([20, 207, 139], dtype="uint8")
        upper = np.array([83,255,255], dtype="uint8")
        # switches(switch)

        # lower = np.array([35, 40, 19], dtype="uint8")
        # upper = np.array([82, 246, 139], dtype="uint8")
        #


        # # switch to HSV

        #green_low_new
        # lower = np.array([17, 34, 18], dtype="uint8")
        # upper = np.array([53, 139, 102], dtype="uint8")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        # #bola azul
        # lower = np.array([55, 80, 26], dtype="uint8")
        # upper = np.array([110, 255, 187], dtype="uint8")

        # find mask of pixels within HSV range
        mask = cv2.inRange(hsv, lower, upper)

        # cv2.imshow("Original", frame)
        # # cv2.imshow("HSV", hsv)
        # cv2.imshow("skinMask", mask)

        # atualizar para a skinmask

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        # mask = cv2.inRange(hsv, greenLower, greenUpper)

        #
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)
        #
        # cv2.imshow("skinMask2", mask)

        #
        #
        # # find contours in the mask and initialize the current
        # # (x, y) center of the ball
        # cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        #                                    cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print(contours)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:
                cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
                print(contours)
                

        cv2.imshow("Original", frame)
        # cv2.imshow("HSV", hsv)
        cv2.imshow("skinMask", mask)


        # print("aqui conntor",cnts)
        # print("aqui hierarquia",hierarchy)
        #
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        # print (cnts)
        # center = None

        # # # only proceed if at least one contour was found
        # if len(cnts) > 0:
        #     # find the largest contour in the mask, then use
        #     # it to compute the minimum enclosing circle and
        #     # centroid
        #     c = max(cnts, key=cv2.contourArea)
        #     ((x, y), radius) = cv2.minEnclosingCircle(c)
        #     M = cv2.moments(c)
        #
        #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #
            # # only proceed if the radius meets a minimum size
            # if radius > 5:
            #     # draw the circle and centroid on the frame,
            #     # then update the list of tracked points
            #     cv2.circle(frame, (int(x), int(y)), int(radius),
            #                (0, 255, 255), 2)
            #     cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # # update the points queue
        # pts.appendleft(center)

        if cv2.waitKey(100) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


# def applyVal(value):
#     print('Applying blur!', value)


if __name__ == '__main__':
   main()
