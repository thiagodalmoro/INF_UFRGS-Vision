import cv2
import numpy as np
import math

NINETY_DEGREES = np.pi/2.0

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


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

    raise Exception('Missing Axis: No perpendicular axis in image')

def slope_intercept_equation(coordinates):
    print(coordinates)
    x1 = coordinates['x1']
    x2 = coordinates['x2']
    y1 = coordinates['y1']
    y2 = coordinates['y2']

    m = (y2 - y1) / (x2 - x1)
    b = -(m * x2) + y2

    # print(coordinates)
    # print('m:',end=' ')
    # print(m)
    # print('b:',end=' ')
    # print(b)
    # print('y2 = x2m + b:')
    # print(str(y2) + ' ' +  '= ' +  str(x2) + ' * ' + str(m) + ' ' + str(b) + ':')
    # print(y2)
    # print(x2*m + b)
    # print()
    #
    # print('m:',end=' ')
    # print(m)
    # print('b:',end=' ')
    # print(b)
    # print('y1 = x1m + b:')
    # print(str(y1) + ' ' +  '= ' +  str(x1) + ' * ' + str(m) + ' + ' + str(b) + ':')
    # print(y1)
    # print(x1*m + b)
    # print()

    print('yLinha = 109m + b:')
    print(str(y1) + ' ' +  '= ' +  str(109) + ' * ' + str(m) + ' + ' + str(b) + ':')
    print(109*m + b)
    print()

    return m,b

def find_perpendicular_equation(m,b,x_intersec,y_intersec):
    new_m = -(1/m)
    new_b = -(new_m * x_intersec) + y_intersec

    return new_m,new_b

#TODO Not working properly
def find_perpendicular_lines(lines):
    axis_equations = []
    adjusted_axis = []
    for line_coords in lines:
        m,b = slope_intercept_equation(line_coords)
        axis_equations.append([m,b])
    print(axis_equations)
    # print(lines[0])
    line1_point1 = [lines[0]['x1'], lines[0]['y1']]
    line1_point2 = [lines[0]['x2'], lines[0]['y2']]
    line2_point1 = [lines[1]['x1'], lines[1]['y1']]
    line2_point2 = [lines[1]['x2'], lines[1]['y2']]
    x_intersec,y_intersec = line_intersection((line1_point1, line1_point2), (line2_point1, line2_point2))
    print('pretty intersec')
    print(line_intersection((line1_point1, line1_point2), (line2_point1, line2_point2)))
    print('ax+b da 1a reta no x_intersec:')
    print(axis_equations[0][0] * x_intersec + axis_equations[0][1])
    print('y_intersec:')
    print(y_intersec)

    m,b = find_perpendicular_equation(axis_equations[1][0],axis_equations[1][1],x_intersec,y_intersec)

    print('axis_equations')
    print(axis_equations)
    print('[m,b]')
    print([m,b])

    x1 = int(-1100)
    y1 = int(x1 * m + b)

    x2 = int(1100)
    y2 = int(x1 * m + b)

    adjusted_axis.append({'x1':x1,'x2':x2,'y1':y1,'y2':y2})
    x1 = int(-1100)
    y1 = int(x1 * axis_equations[1][0] + axis_equations[1][1])

    x2 = int(1100)
    y2 = int(x2 * axis_equations[1][0] + axis_equations[1][1])

    adjusted_axis.append({'x1':x1,'x2':x2,'y1':y1,'y2':y2})

    return adjusted_axis


def find_axis(image):

    #Apply Hough Transform (that returns the vector of lines) into the canny edge detection image
    accumulator_threshold = 150 # Threshold to consider a line ("votes necessaries to be considered line")
    hough_lines = cv2.HoughLines(image, 1, np.pi / 180, accumulator_threshold)

    axis_lines = find_axis_lines(hough_lines)

    lines_coordinates = translate_polar_coordinates(image, axis_lines)

    #TODO return adjusted_axis when find_perpendicular_lines properly implemented
    adjusted_axis = find_perpendicular_lines(lines_coordinates)

    return lines_coordinates

def draw_lines(src_image, lines):
    for line in lines:
        # print(line)
        cv2.line(src_image, ( line['x1'],  line['y1']), ( line['x2'],  line['y2']), (255, 0, 0), 2)

def find_and_draw_axis(src_image):
    processed_image = process_image(src_image)
    axis = find_axis(processed_image)
    draw_lines(src_image, axis)

    return src_image

# main()
if __name__ == '__main__' :

    img1 = cv2.imread('exemplo3.jpg')
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
