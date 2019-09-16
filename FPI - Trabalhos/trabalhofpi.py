#Trabalho FPI
#Thiago Dal Moro


import cv2
import numpy as np
import tkinter as tk
from PIL import Image


#imagem 1 - Original
img = cv.imread('Gramado_22k.jpg',1)
cv.imshow('Image', img)

altura, largura = img.shape[:2]
cv.waitKey(0)

if k == 27:         # wait for ESC key to exit
     cv.destroyAllWindows()
 elif k == ord('s'): # wait for 's' key to rotate
    #rotacao
    ponto = (largura / 2, altura / 2) #ponto no centro da figura
    rotacao = cv2.getRotationMatrix2D(ponto, 45, 1.0)
    rotacionado = cv2.warpAffine(imagem, rotacao, (largura, altura))
    cv2.imshow("Rotacionado 45 graus", rotacionado)

    cv2.waitKey(0)

    rotacao = cv2.getRotationMatrix2D(ponto, 120, 1.0)
    rotacionado = cv2.warpAffine(imagem, rotacao, (largura, altura))
    cv2.imshow("Rotacionado 120 graus", rotacionado)

    cv2.waitKey(0)






#
#
# #translacao (deslocamento)
# deslocamento = np.float32([[1, 0, 25], [0, 1, 50]])
# deslocado = cv2.warpAffine(imagem, deslocamento, (largura, altura))
# cv2.imshow("Baixo e direita", deslocado)
#
# cv2.waitKey(0)
#
# deslocamento = np.float32([[1, 0, -50], [0, 1, -90]])
# deslocado = cv2.warpAffine(imagem, deslocamento, (largura, altura))
# cv2.imshow("Cima e esquerda", deslocado)
#
# cv2.waitKey(0)
#
#
#


# Imagem 2 - Alterada
img2 = cv.imread('Gramado_72k.jpg',0)
cv.imshow('Image Alterada', img2)



# k = cv.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv.imwrite('Gramado_22k(2).png',img)


cv.destroyAllWindows()
