import cv2
import numpy as np

def translate_polar_coordinates (src_image, hough_lines):
    editing_image = src_image
    # Calulate the image's larger measure (lenght or width) to fit the lines we will draw in the image
    img_max_size = max(src_image.shape) * 1.1

    lines_list = []
    # We iterate over the hough_lines
    for line in hough_lines[0:-1]:
        #We want to transform polar coordinates to "normal" coordinates
        for rho, theta in line:
            cosTheta = np.cos(theta)
            sinTheta = np.sin(theta)
            x0 = cosTheta * rho
            y0 = sinTheta * rho

            #Calculate coordinates of the first point: r * cos(theta) - img_max_size * [sin(theta) or cos(theta)]
            x1 = int(x0 + img_max_size * (-sinTheta)) # ( r * cos(theta) - img_max_size * sin(theta))
            y1 = int(y0 + img_max_size * (cosTheta))  # ( r * sin(theta) + img_max_size * cos(theta))

            #Calculate coordinates of the second point
            x2 = int(x0 - img_max_size * (-sinTheta)) # ( r * cos(theta) + img_max_size * sin(theta))
            y2 = int(y0 - img_max_size * (cosTheta))  # ( r * sin(theta) - img_max_size * cos(theta))
            coordinates = {'x1':x1,'x2':x2,'y1':y1,'y2':y2}
            lines_list.append(coordinates)

    return lines_list

def process_image(image):
    # Convert into gray scale image (for canny edge detection is preferred gray scale images)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Process Edge Detection (gets the canny edge detection image)
    lower_threshold = 100
    higher_threshold = 200
    edges = cv2.Canny(gray, lower_threshold, higher_threshold, apertureSize = 3)

    return edges

def find_lines(image):

    #Apply Hough Transform (that returns the vector of lines) into the canny edge detection image
    accumulator_threshold = 150 # Threshold to consider a line ("votes necessaries to be considered line")
    hough_lines = cv2.HoughLines(image, 1, np.pi / 180, accumulator_threshold)

    lines = translate_polar_coordinates(image, hough_lines)

    return lines

def draw_lines(src_image, lines):
    for line in lines:
        cv2.line(src_image, ( line['x1'],  line['y1']), ( line['x2'],  line['y2']), (0, 0, 255), 2)

def find_and_draw_axis(src_image):
    processed_image = process_image(src_image)
    lines = find_lines(processed_image)
    draw_lines(src_image, lines)

    return src_image

# main()
if __name__ == '__main__' :

    img1 = cv2.imread('t.jpg')
    img1 = find_and_draw_axis(img1)
    cv2.imshow('Image1', img1)

    # img2 = cv2.imread('exemplo2.jpg')
    # img2 = find_and_draw_axis(img2)
    # cv2.imshow('Image2', img2)
    #
    # img3 = cv2.imread('exemplo3.jpg')
    # img3 = find_and_draw_axis(img3)
    # cv2.imshow('Image3', img3)
    # cv2.destroyAllWindows()

    key = cv2.waitKey(0)
    if key == 27:  # wait for ESC
        print('Key ESC pressed.')
        cv2.destroyAllWindows()
