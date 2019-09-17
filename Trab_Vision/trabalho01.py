#Trabalho Computer Vision
#Thiago Dal Moro


import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import Image




def mouse_drawing(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print("Left click")
        cv.line(img1,(0,0),(511,511),(255,0,0),5)

        circles.append((x, y))
        print(circles)

    # # Draw a diagonal blue line with thickness of 5 px




#imagem 1 - Original
img1 = cv.imread('maracana1.jpg',1)
cv.imshow('Image1', img1)
cv.namedWindow("Image1")
cv.setMouseCallback("Image1", mouse_drawing)


circles = []
while True:

    for center_position in circles:
        cv.circle(img1, center_position, 5, (0, 0, 255), -1)
    cv.imshow("Image1", img1)


    key= cv.waitKey(0)
    if key == 27:         # wait for ESC key to exit
        print('Key ESC pressed.')
        cv.destroyAllWindows()

    elif key == ord("d"):
        circles = []



        #rotacao
        # ponto = (largura / 2, altura / 2) #ponto no centro da figura
        # rotacao = cv2.getRotationMatrix2D(ponto, 45, 1.0)
        # rotacionado = cv2.warpAffine(imagem, rotacao, (largura, altura))
        # cv2.imshow("Rotacionado 45 graus", rotacionado)
        #
        # cv2.waitKey(0)
        #
        # rotacao = cv2.getRotationMatrix2D(ponto, 120, 1.0)
        # rotacionado = cv2.warpAffine(imagem, rotacao, (largura, altura))
        # cv2.imshow("Rotacionado 120 graus", rotacionado)

        cv.waitKey(0)
