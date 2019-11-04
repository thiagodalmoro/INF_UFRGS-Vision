import numpy as np
import cv2 as cv
# from matplotlib.pyplot import matplotlib
# import matplotlib.pyplot
# from matplotlib import pyplot as plt

line = []

def mouse_drawing(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print("Left click")
        print(line)

    # # Draw a diagonal blue line with thickness of 5 px





#imagem 1 - Original
img1 = cv.imread('exemplo1.jpg',1)
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

cv.imshow('Image1', img1)

cv.namedWindow("Image1")

cv.imshow ('Image Gray', gray)

# cv.setMouseCallback("Image1", mouse_drawing)
kernel_size = 5
blur_gray = cv.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv.Canny(blur_gray, low_threshold, high_threshold)



# #converted = convert_hls(img)
# image = cv.cvtColor(img1, cv.COLOR_BGR2HLS)
# lower = np.uint8([0, 200, 0])
# upper = np.uint8([255, 255, 255])
# white_mask = cv.inRange(image, lower, upper)
# yellow color mask
# lower = np.uint8([10, 0,   100])
# upper = np.uint8([40, 255, 255])
# yellow_mask = cv.inRange(image, lower, upper)
# # combine the mask
# mask = cv.bitwise_or(white_mask, yellow_mask)
# result = img1.copy()
# cv.imshow("mask",mask)




#pixel exmplo 1 = alto [220, 61], baixo [235, 660]  = y
#pixel exmplo 1 = esq [54, 548], dir [ 981, 567]  = x

start_pointx = (54, 548)
end_pointx = (981, 567)


start_pointy = (220, 61)
end_pointy = (235, 660)

#color = green
color = (0,255,0)

#pxl px
thinckness = 1

#
# # # Draw a diagonal blue line with thickness of 5 px
# img = cv.line(img1,start_pointy ,end_pointy,color,thinckness)
# cv.imshow('ImageDraw', img)


# # # Draw a diagonal blue line with thickness of 5 px
# img2 = cv.line(img1,start_pointx ,end_pointx,color,thinckness)
# cv.imshow('ImageDraw', img)




key= cv.waitKey(0)
if key == 27:         # wait for ESC key to exit
    print('Key ESC pressed.')
    cv.destroyAllWindows()
