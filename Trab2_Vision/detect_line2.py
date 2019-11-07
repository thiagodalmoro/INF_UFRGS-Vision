import cv2
import numpy as np

# TODO: check if this is the proper name for this function
def Hough (image, list):
    print('list param entrada:')
    for l in list:
        print(l)
    print('')
    for line in list:
        #We want to transform polar coordinates to "normal" coordinates
        for rho, theta in line:
            print(line)
            print()
            cosTheta = np.cos(theta)
            sinTheta = np.sin(theta)
            x0 = cosTheta * rho
            y0 = sinTheta * rho

            #Calculate coordintes of the first point
            x1 = int(x0 + 1100 * (-sinTheta)) # ( r * cos(theta) - 1100 sin(theta))
            y1 = int(y0 + 1100 * (cosTheta))  # ( r * sin(theta) + 1100 cos(theta))

            #Calculate coordintes of the second point
            x2 = int(x0 - 1100 * (-sinTheta)) # ( r * cos(theta) + 1100 sin(theta))
            y2 = int(y0 - 1100 * (cosTheta))  # ( r * sin(theta) - 1100 cos(theta))

            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)



def find_and_draw_axis(src_image):
    editing_image = src_image

    # Convert into gray scale image (for canny edge detection is preferred gray scale images)
    gray = cv2.cvtColor(editing_image,cv2.COLOR_BGR2GRAY)

    # Process Edge Detection (gets the canny edge detection image)
    lower_threshold = 50
    higher_threshold = 150
    edges = cv2.Canny(gray, lower_threshold, higher_threshold, apertureSize = 3)

    #Apply Hough Transform (that returns the vector of lines) into the canny edge detection image
    accumulator_threshold = 180 # Threshold to consider a line
    lines = cv2.HoughLines(edges, 1, np.pi / 180, accumulator_threshold, None, 0, 0)
    print(lines)

    # TODO: check how to find 2 proper lines
    ex1 = lines[0::1]
    # ex1.append(lines[0])
    # ex1.append(lines[2])

    Hough(editing_image,lines)

    return editing_image



# main()
if __name__ == '__main__' :

    img1 = cv2.imread('exemplo1.jpg')
    img1 = find_and_draw_axis(img1)
    cv2.imshow('Image1', img1)

    img2 = cv2.imread('exemplo2.jpg')
    img2 = find_and_draw_axis(img2)
    cv2.imshow('Image2', img2)

    img3 = cv2.imread('exemplo3.jpg')
    img3 = find_and_draw_axis(img3)
    cv2.imshow('Image3', img3)
    # cv2.destroyAllWindows()

    key = cv2.waitKey(0)
    if key == 27:  # wait for ESC
        print('Key ESC pressed.')
        cv2.destroyAllWindows()
