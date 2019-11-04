import numpy as np
import cv2 as cv

line = []

def mouse_drawing(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print("Left click")
        print(line)

    # # Draw a diagonal blue line with thickness of 5 px





#imagem 1 - Original
img1 = cv.imread('exemplo1.jpg',1)
cv.imshow('Image1', img1)
cv.namedWindow("Image1")
cv.setMouseCallback("Image1", mouse_drawing)

#pixel exmplo 1 = alto [220, 61], baixo [235, 660]  = y
#pixel exmplo 1 = esq [54, 548], dir [ 981, 567]  = x


# # Draw a diagonal blue line with thickness of 5 px
# img = cv.line(img1,(line[0],line[1]),(511,511),(0,0,255),2)
# cv.imshow('Image1', img)



key= cv.waitKey(0)
if key == 27:         # wait for ESC key to exit
    print('Key ESC pressed.')
    cv.destroyAllWindows()
