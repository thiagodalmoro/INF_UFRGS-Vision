#
# name: main-3rd.py
# Author: Pedro Cantarutti
#

from tkinter import X
from tkinter.ttk import Style
#import PIL.Image, PIL.ImageTk
from PIL import Image, ImageEnhance
import tkinter as tk
import numpy as np
import cv2
import math


def switches(switch):
    cv2.createTrackbar(switch, 'Video Frame', 0, 1, applyVal)
    cv2.createTrackbar('lower', 'Video Frame', 0, 255, applyVal)
    cv2.createTrackbar('upper', 'Video Frame', 0, 255, applyVal)
    cv2.createTrackbar('Blur', 'Video Frame', 3, 5, applyVal)


def rotate_image(frame, angle):
    height, width = frame.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_frame = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_frame[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_frame[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_frame = cv2.warpAffine(frame, rotation_frame, (bound_w, bound_h))
    return rotated_frame


# TODO(pac): revise
def apply_contrast(frame, level):
    cont = ImageEnhance.Contrast(frame)
    cont = cont.enhance(level)
    enh_frame = np.array(cont.getdata())
    return enh_frame


def apply_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray frame
    return gray


def apply_blur(frame, kernel):
    blur = cv2.GaussianBlur(frame,(3,3),0) # blur the video fram by some value
    return blur

def flip_frame(frame, code):
    flipped_frame = cv2.flip(frame, flipCode=code)
    return flipped_frame


def detect_edges(frame, lower, upper, s):
    if s == 0:
        edges = frame
    else:
        edges = cv2.Canny(frame, lower, upper)
    return edges


def apply_rescale(frame, percent):
    frame_width = int(frame.shape[1] * percent/ 100)
    frame_height = int(frame.shape[0] * percent/ 100)
    dim = (frame_width, frame_height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized_frame


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original Output')

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 17.0, (int(frame_width),int(frame_height)))

    switch = '0 : OFF \n1 : ON'
    switches(switch)

    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        lower = cv2.getTrackbarPos('lower', 'Original Output')
        upper = cv2.getTrackbarPos('upper', 'Original Output')
        s = cv2.getTrackbarPos(switch, 'Original Output')

        edged_frame = detect_edges(frame, lower, upper, s)

        sobelx_frame = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
        sobelx_64f = np.absolute(sobelx_frame)
        sobelx_8u_frame = np.uint8(sobelx_64f)

        sobely_frame = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
        sobely_64f = np.absolute(sobely_frame)
        sobely_8u_frame = np.uint8(sobely_64f)

        rotated90_frame = rotate_image(frame, 90)
        flipped_v_frame = flip_frame(frame, 1) # vertical
        flipped_h_frame = flip_frame(frame, 0) # horizontal

        gray_frame = apply_grayscale(frame)
        gau_blur_frame = apply_blur(frame, (3,3))
        resized_frame = apply_rescale(frame, 50) # downscale

        cv2.imshow('Original Output', frame)
        cv2.imshow('GaussianBlur', gau_blur_frame)
        cv2.imshow('Canny', edged_frame)
        cv2.imshow('SobelX', sobelx_8u_frame)
        cv2.imshow('SobelY', sobely_8u_frame)
        cv2.imshow('Grayscale', gray_frame)
        cv2.imshow('Rotated 90 degrees', rotated90_frame)
        cv2.imshow('flip v', flipped_v_frame)
        cv2.imshow('flip h', flipped_h_frame)
        cv2.imshow('Rescale', resized_frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def applyVal(value):
    print('Applying blur!', value)


if __name__ == '__main__':
   main()
