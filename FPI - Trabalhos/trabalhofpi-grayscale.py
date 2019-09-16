#Trabalho FPI
#Thiago Dal Moro

import cv2
import numpy as np
import tkinter as tk
from PIL import Image


#imagem 1 - Original

imagem = cv2.imread('Gramado_22k.jpg',1)
cv2.imshow('Imagem Original', imagem)

k= cv2.waitKey(0)

if k == 27:         # wait for ESC key to exit
    print('Key ESC pressed.')
    cv.destroyAllWindows()

elif k == ord('s'): # wait for 's' key to grayscale

    #grayscale
     gray_image = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
     cv2.imshow('Escala de Cinza', gray_image)
     print('Gray Image.')

k= cv2.waitKey(0)
