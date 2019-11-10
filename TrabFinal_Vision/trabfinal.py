#
# Trabalho Final - Cadeira FPI



from tkinter import X
from tkinter.ttk import Style
import PIL.Image, PIL.ImageTk
import tkinter as tk
import numpy as np
import cv2

def switches(switch):
    cv2.createTrackbar(switch, 'Video Frame', 0, 1, applyVal)
    cv2.createTrackbar('lower', 'Video Frame', 0, 255, applyVal)
    cv2.createTrackbar('upper', 'Video Frame', 0, 255, applyVal)
    cv2.createTrackbar('Blur', 'Video Frame', 3, 5, applyVal)




def main():
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Video Frame')

    switch = '0 : OFF \n1 : ON'

    switches(switch)
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray frame
        blur = cv2.GaussianBlur(frame,(3,3),0) # blur the video fram by some value

        # get current positions of four trackbars
        lower = cv2.getTrackbarPos('lower', 'Video Frame')
        upper = cv2.getTrackbarPos('upper', 'Video Frame')
        s = cv2.getTrackbarPos(switch, 'Video Frame')

        if s == 0:
            edges = frame
        else:
            edges = cv2.Canny(frame, lower, upper)

        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

        # display images
        cv2.imshow('original', frame)
        #cv2.imshow('SobelX', sobelx)
        #cv2.imshow('SobelY', sobely)
        cv2.imshow('Video Frame', edges)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def applyVal(value):
    print('Applying blur!', value)


if __name__ == '__main__':
   main()
