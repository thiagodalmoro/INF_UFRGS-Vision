#Trabalho FPI
#Thiago Dal Moro

import cv2
import numpy as np
import tkinter as tk
from PIL import Image


#imagem 1 - Original

imagem = cv2.imread('Gramado_22k.jpg',1)
cv2.imshow('Imagem Original', imagem)

k = cv2.waitKey(0)

if k == 27:         # wait for ESC key to exit
    print('Key ESC pressed.')
    cv2.destroyAllWindows()

elif k == ord('s'): # wait for 's' key to horizontal


    altura, largura = imagem.shape[:2]

    q = 16
    r = 256 / q # Ensure your image is on [0 255] range

    for row in range(0, largura):
        for col in range(0, altura):
            imagem[col][row] = abs(imagem[col][row] / r)

    cv2.imshow( "Quantized 16x", imagem )
    print('Quantized 16x Image loaded.')












cv2.waitKey(0)

cv.destroyAllWindows()
