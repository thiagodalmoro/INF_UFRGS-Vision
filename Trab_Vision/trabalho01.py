#Trabalho Computer Vision
#Thiago Dal Moro


import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import Image


#imagem 1 - Original
img1 = cv.imread('maracana1.jpg',1)
cv.imshow('Image1', img1)

#imagem 2 - Original
img2 = cv.imread('maracana2.jpg',1)
cv.imshow('Image2 ', img2)

alt1, lar1 = img1.shape[:2]
alt2, lar2 = img2.shape[:2]


#to quit
k = cv.waitKey(0)

if k == 27:         # wait for ESC key to exit
    print('Key ESC pressed.')
    cv.destroyAllWindows()

elif k == ord('s'): # wait for 's' key to rotate


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

    cv2.waitKey(0)
