import cv2
import numpy as np

NINETY_DEGREES = np.pi/2.0

def translate_polar_coordinates (image, axis_lines):
    # Calulate the image larger measure (lenght or width) to fit the lines we will draw in the image
    img_max_size = max(image.shape) * 1.1

    lines_list = []
    # We iterate over the axis_lines
    for line in axis_lines:
        #We want to transform polar coordinates to "normal" coordinates
        # for rho, theta in line:
        rho = line[0]
        theta = line[1]
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
    edge_map_image = cv2.Canny(gray, lower_threshold, higher_threshold, apertureSize = 3)

    #Dilate edge map to deal with eventual holes in the middle of the lines
    kernel = np.ones((5,5), np.uint8)
    dilated_image = cv2.dilate(edge_map_image, kernel, iterations=1)
    cv2.imshow('canny1', dilated_image)

    return dilated_image

def find_axis_lines(lines_list):
    axis = []
    angle_threshold = np.pi / 16

    #The "most voted" line is one of the axis
    axis.append([lines_list[0][0][0],lines_list[0][0][1]])

    #The other axis will be the first line that forms an angle of 90 with the first axis
    # Will be considered a margin of error to not be needed exactly 90 degrees
    for line in lines_list:
        for ro,theta in line:
            #We want 2 lines with an angle of 90 degrees and a margin of error of pi/16
            # We calculate the difference between the angles
            angle_difference = abs(axis[0][1] - theta)
            #Then the difference minus 90 degress (pi/2)
            normalized_angle_difference = abs(angle_difference - NINETY_DEGREES )
            #If the angle was 90 degrees, we will have 0, if is somehting less than our margin/threshold we accept it too
            if normalized_angle_difference < angle_threshold:
                axis.append([ro,theta]) # Append the other axis coordinates
                return axis

    raise Exception('Missing Axis')

def find_axis(image):

    #Apply Hough Transform (that returns the vector of lines) into the canny edge detection image
    accumulator_threshold = 150 # Threshold to consider a line ("votes necessaries to be considered line")
    hough_lines = cv2.HoughLines(image, 1, np.pi / 180, accumulator_threshold)

    axis_lines = find_axis_lines(hough_lines)


    lines = translate_polar_coordinates(image, axis_lines)


    return lines

def draw_lines(src_image, lines):
    for line in lines:
        cv2.line(src_image, ( line['x1'],  line['y1']), ( line['x2'],  line['y2']), (255, 0, 0), 2)

def find_and_draw_axis(src_image):
    processed_image = process_image(src_image)
    axis = find_axis(processed_image)
    draw_lines(src_image, axis)

    return src_image

# main()
if __name__ == '__main__' :

    img1 = cv2.imread('tst.png')
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
