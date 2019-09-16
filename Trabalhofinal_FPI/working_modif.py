# Trabalho 03 - FPI
#
# Thiago Dal Moro
#
#
#criação de videos e frames
#

import numpy as np
import cv2
import math
from tkinter import X
from tkinter.ttk import Style
import PIL.Image, PIL.ImageTk
from PIL import Image, ImageEnhance
import tkinter as tk


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

def nothing(x):
    pass




def switches(switch):
    cv2.createTrackbar(switch, 'Video Frame', 0, 1, nothing)
    cv2.createTrackbar('H_MIN', 'Video Frame', H_MIN, H_MAX, nothing)
    cv2.createTrackbar('H_MAX', 'Video Frame', H_MAX, H_MAX, nothing)
    cv2.createTrackbar('S_MIN', 'Video Frame', S_MIN, S_MAX, nothing)
    cv2.createTrackbar('S_MAX', 'Video Frame', S_MAX, S_MAX, nothing)
    cv2.createTrackbar('V_MIN', 'Video Frame', V_MIN,  V_MAX, nothing)
    cv2.createTrackbar('V_MAX', 'Video Frame', V_MAX, V_MAX, nothing)



# def switches(switch):
#     cv2.createTrackbar(switch, 'Video Frame', 0, 1, app_cota)
#     cv2.createTrackbar('lower', 'Video Frame', 0, 255, app_cota)
#     cv2.createTrackbar('upper', 'Video Frame', 0, 255, app_cota)
#     cv2.createTrackbar('GaussianBlur', 'Video Frame', 3, 5, app_cota)

def rescale (frame, percent):

    frame_width = int(frame.shape[1] * percent/ 100)
    frame_height = int(frame.shape[0] * percent/ 100)
    tamanho = (frame_width, frame_height)
    frame_rescale = cv2.resize(frame, tamanho, interpolation=cv2.INTER_AREA)

    return frame_rescale

def main():

    cap = cv2.VideoCapture(0)

    #gravacao do video
    #criacao do arquivo de video
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (int(frame_width),int(frame_height)))

    cv2.namedWindow('Video Frame')

    switch = ' 0 : OFF \n    1 : ON'

    switches(switch)

    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        #trackbars


        # get current positions of four trackbars
        hmin = cv2.getTrackbarPos('H_MIN','Video Frame')
        hmax = cv2.getTrackbarPos('H_MAX','Video Frame')
        vmin = cv2.getTrackbarPos('S_MIN','Video Frame')
        vmax = cv2.getTrackbarPos('S_MAX','Video Frame')
        smin = cv2.getTrackbarPos('S_MIN','Video Frame')
        smax = cv2.getTrackbarPos('S_MAX','Video Frame')

        s = cv2.getTrackbarPos(switch,'Video Frame')

        # lower = cv2.getTrackbarPos('lower', 'Video Frame')
        # upper = cv2.getTrackbarPos('upper', 'Video Frame')
        # sinal = cv2.getTrackbarPos(switch, 'Video Frame')

        frame_rescale = rescale (frame, 50)


        lower = np.array([hmin, vmin, smin], dtype="uint8")
        upper = np.array([hmax,vmax,smax], dtype="uint8")

        # switch to HSV
        hsv = cv2.cvtColor(frame_rescale, cv2.COLOR_BGR2HSV)

        # find mask of pixels within HSV range
        mask = cv2.inRange(hsv, lower, upper)

        # denoise
        mask1 = cv2.GaussianBlur(mask, (9, 9), 0)

        # kernel for morphology operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

        # CLOSE (dilate / erode)
        mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations = 3)

        # only display the masked pixels
        skin = cv2.bitwise_and(frame_rescale, frame_rescale, mask = mask2)

        cv2.imshow("HSV", hsv)
        cv2.imshow('Video Frame', frame_rescale)
        cv2.imshow('Mask ', mask)
        cv2.imshow('Mask1', mask1)
        cv2.imshow('Mask2 ', mask2)
        cv2.imshow('Skin ', skin)



#onde parei?

#preciso ainda salvar os valores
#passar para uma Matrix
#salvar a posição do object
#desenhar quadrado, circulo







    # condição para saida do programa
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break



    cap.release()
    # out.release()
    cv2.destroyAllWindows()

# def app_cota (value):
#     #
#     # if (lower != 0)
#         print('Valor:', value)



if __name__ == '__main__':
   main()
