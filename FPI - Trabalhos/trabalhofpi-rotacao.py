#Trabalho FPI
#Thiago Dal Moro

import cv2
import numpy as np
import tkinter as tk
from PIL import Image


#imagem 1 - Original

imagem = cv2.imread('Gramado_22k.jpg',1)
cv2.imshow('Imagem Original', imagem)

altura, largura = imagem.shape[:2]

k= cv2.waitKey(0)

if k == 27:         # wait for ESC key to exit
    print('Key ESC pressed.')
    cv.destroyAllWindows()

elif k == ord('1'): # wait for '2' key to rotate
    #rotacao
    ponto = (largura / 2, altura / 2) #ponto no centro da figura
    rotacao = cv2.getRotationMatrix2D(ponto, 45, 1.0)
    rotacionado = cv2.warpAffine(imagem, rotacao, (largura, altura))
    cv2.imshow("Rotacionado 45 graus", rotacionado)
    print('Imagem rotacionada 45 graus.')

k1= cv2.waitKey(0)

if k1 == 27:         # wait for ESC key to exit
    print('Key ESC pressed.')
    cv.destroyAllWindows()
       

elif k1 == ord('2'): # wait for '2' key to rotate
        rotacao = cv2.getRotationMatrix2D(ponto, 120, 1.0)
        rotacionado = cv2.warpAffine(imagem, rotacao, (largura, altura))
        cv2.imshow("Rotacionado 120 graus", rotacionado)
        print('Imagem rotacionada 120 graus.')

        cv2.waitKey(0)







#


# k = cv.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv.imwrite('Gramado_22k(2).png',img)


cv.destroyAllWindows()
