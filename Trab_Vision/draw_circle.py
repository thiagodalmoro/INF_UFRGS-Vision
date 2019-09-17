import cv2
import numpy as np

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:

        rho = 1
        theta = np.pi/180

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho


        # cv2.line(img1,(x,y),(x0, -y0),(0,0,255),2)
        print("Left click")
        print(x,y)
        print(a,b)
        print(x0,y0)




    # a = np.cos(theta)
    # b = np.sin(theta)
    # x0 = a*rho
    # y0 = b*rho
    # # x1 = int(x0 + 1000*(-b))
    # # y1 = int(y0 + 1000*(a))

#
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#
# cv2.imwrite('houghlines3.jpg',img)
#



# Create a black image, a window and bind the function to window
#imagem 1 - Original
img1 = cv2.imread('maracana1.jpg',1)
cv2.imshow('Image1', img1)
cv2.namedWindow("Image1")
cv2.setMouseCallback("Image1", draw_circle)


key= cv2.waitKey(0)

while(1):
    cv2.imshow('Image1',img1)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
