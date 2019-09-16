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

elif k == ord('s'): # wait for 's' key to horizontal

    flip_img = cv2.flip( imagem, 1 )
    cv2.imshow( "Horizontal flip", flip_img )
    print('Horizontal Flipped Image loaded.')
    cv2.imwrite('h-flipped-img.jpeg', flip_img)
    print('Horizontal Flipped Image saved.')

cv2.waitKey(0)

cv.destroyAllWindows()
