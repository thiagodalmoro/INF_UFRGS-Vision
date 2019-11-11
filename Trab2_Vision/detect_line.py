# Python code to detect an arrow (seven-sided shape) from an image.
import numpy as np
import cv2 as cv


# Reading image
img = cv.imread('exemplo1.jpg', cv.IMREAD_COLOR)

# Reading same image in another variable and
# converting to gray scale.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kernel_size = 7
blur_gray = cv.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150

edges = cv.Canny(blur_gray, low_threshold, high_threshold)

lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1,y1), (x2,y2), (0,255,0), 2)




cv.imshow('Image1', img)
cv.imshow ('Image Gray', gray)
cv.imshow ('Image Edges', edges)




# start_pointx = (54, 548)
# end_pointx = (981, 567)
#
#
# start_pointy = (220, 61)
# end_pointy = (235, 660)
#

key= cv.waitKey(0)
if key == 27:         # wait for ESC key to exit
    print('Key ESC pressed.')
    cv.destroyAllWindows()
