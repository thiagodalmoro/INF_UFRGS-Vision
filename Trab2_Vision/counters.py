import numpy as np
import cv2

im2 = cv2.imread('exemplo1.jpg',1)
imgray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('im2', im2)
cv2.imshow('gray', imgray)
cv2.imshow('ret ', ret )
cv2.imshow('Th',thresh)

cv2.drawContours(ret, contours, -1, (0,255,0), 3)

cv2.drawContours(ret, contours, 3, (0,255,0), 3)


cnt = contours[4]
cv2.drawContours(ret, [cnt], 0, (0,255,0), 3)

key= cv2.waitKey(0)
if key == 27:         # wait for ESC key to exit
    print('Key ESC pressed.')
    cv2.destroyAllWindows()




