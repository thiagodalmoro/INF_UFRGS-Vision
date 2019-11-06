import cv2
import numpy as np

def Hough (image, list):

    for i in range(2):
        # print(i)
        for rho, theta in list[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


img1 = cv2.imread('exemplo1.jpg')
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
print(lines)

ex1 = []
ex1.append(lines[0])
ex1.append(lines[2])

Hough(img1,ex1)

img2 = cv2.imread('exemplo2.jpg')
gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
print(lines)

ex2 = []
ex2.append(lines[0])
ex2.append(lines[2])

Hough(img2,ex2)

img3 = cv2.imread('exemplo3.jpg')
gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
print(lines)

ex3 = []
ex3.append(lines[3])
ex3.append(lines[2])

Hough(img3,ex3)

cv2.imshow('Image1', img1)
cv2.imshow('Image2', img2)
cv2.imshow('Image3', img3)

key = cv2.waitKey(0)
if key == 27:  # wait for ESC
    print('Key ESC pressed.')
    cv2.destroyAllWindows()
