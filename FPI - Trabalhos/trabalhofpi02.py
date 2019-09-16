#Trabalho FPI
#Thiago Dal Moro
#Parte II – Leitura, Exibição e Operações sobre Imagens (80 pontos)


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#imagem 1 - Original
img = cv.imread('Gramado_22k.jpg',1)
cv.imshow('Image', img)


# Imagem 2 - Alterada
img2 = cv.imread('Gramado_72k.jpg',0)
cv.imshow('Image Alterada', img2)


k = cv.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv.imwrite('Gramado_22k(2).png',img)
    print ('Imamge salva com sucesso.')
    
