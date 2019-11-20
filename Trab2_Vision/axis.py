import cv2
import numpy as np
import math


NINETY_DEGREES = np.pi/2.0

def remove_small_components(img):
    #find all connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 500

    # Answer image
    img2 = np.zeros((output.shape))
    #for every component in the image,  keep only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2

def calculate_line_intersection(line1, line2):
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
    #Since edge detection is susceptible to noise in the image, first step is to remove the noise in the image with a 5x5 Gaussian filter.
    blur = cv2.GaussianBlur(image,(5,5),0)

    # Convert into gray scale image (for canny edge detection is preferred gray scale images)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

    # Process Edge Detection (gets the canny edge detection image)
    lower_threshold = 50
    higher_threshold = 200
    edge_map_image = cv2.Canny(gray, lower_threshold, higher_threshold, apertureSize = 3)

    #Dilate edge map to deal with eventual holes in the middle of the lines
    kernel = np.ones((5,5), np.uint8)
    dilated_image = cv2.dilate(edge_map_image, kernel, iterations=1)

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
    x1 = coordinates['x1']
    x2 = coordinates['x2']
    y1 = coordinates['y1']
    y2 = coordinates['y2']

    m = (y2 - y1) / (x2 - x1)
    b = -(m * x2) + y2

    return m,b

def find_perpendicular_equation(m,b,x_intersec,y_intersec):
    new_m = -(1/m)
    new_b = -(new_m * x_intersec) + y_intersec

    return new_m,new_b

def find_line_intersection(lines):
    line1_point1 = [lines[0]['x1'], lines[0]['y1']]
    line1_point2 = [lines[0]['x2'], lines[0]['y2']]
    line2_point1 = [lines[1]['x1'], lines[1]['y1']]
    line2_point2 = [lines[1]['x2'], lines[1]['y2']]
    x_intersec,y_intersec = calculate_line_intersection((line1_point1, line1_point2), (line2_point1, line2_point2))

    return x_intersec,y_intersec

def find_lines_equation(lines):
    axis_equations = []
    for line_coords in lines:
        m,b = slope_intercept_equation(line_coords)
        axis_equations.append([m,b])

    return axis_equations

def find_axis_limits(m,b):
    x1 = int(-1100)
    y1 = int(x1*m + b)

    x2 = int(1100)
    y2 = int(x2*m + b)

    return {'x1':x1,'x2':x2,'y1':y1,'y2':y2}

def adjust_axis_lines(m1,b1,m2,b2):
    adjusted_axis = []
    x_axis = find_axis_limits(m1,b1)
    y_axis = find_axis_limits(m2,b2)
    adjusted_axis.append(x_axis)
    adjusted_axis.append(y_axis)

    return adjusted_axis

def angle_between_lines(m1,b1,m2,b2):
    try:
        division = (m2 - m1) / (1 + m2*m1)
    except:
        return 0
    angle = np.arctan(division)
    return angle

def calculate_angle_direction(m,b,angle):
    #The X axis should be a horizontal straight line with a constant y
    y0 = 0 * m + b
    y1 = 1100 * m + b

    #If Y increases it's because it has positive rotation
    if y1 - y0 > 0:
        angle = angle

    #If Y decreases it's because it has negative rotation
    if y1 - y0 < 0:
        print('aqui')
        angle = -angle

    print(angle)
    return angle

def find_axis_rotation(m1,b1,m2,b2):
    m_horizontal = 0
    b_horizontal = 1

    #Angle between both lines and a horizontal straight line
    angle1 = abs(angle_between_lines(m1,b1,m_horizontal,b_horizontal))
    angle2 = abs(angle_between_lines(m2,b2,m_horizontal,b_horizontal))

    #Calculate which line is the Y axis by comparing how close their angle is to 90 degress
    angle1 = abs(angle1 - NINETY_DEGREES)
    angle2 = abs(angle2 - NINETY_DEGREES)

    #Once we know which is the Y axis, we check the direction of the rotation analising X axis
    if angle1 < angle2:
        angle_rotation = calculate_angle_direction(m2,b2,angle1)

    else:
        angle_rotation = calculate_angle_direction(m1,b1,angle2)

    return angle_rotation

def find_perpendicular_lines(axis_lines):
    #Find axis lines equations (y = mx + b)
    axis_equations = find_lines_equation(axis_lines)

    #Calulate where they intercept each other
    x_intersec,y_intersec = find_line_intersection(axis_lines)

    # Calculate the equation of a line that crosses the first axis and its orthogonal at the point where 2 axis intercept
    m,b = find_perpendicular_equation(axis_equations[0][0],axis_equations[0][1],x_intersec,y_intersec)

    #As we were considering lines as a pair of 2 points/coordinates, we want both equations again in pair of points
    adjusted_axis = adjust_axis_lines(axis_equations[0][0],axis_equations[0][1],m,b)

    # Consider also the rotation the axis in the image
    angle_rotation = find_axis_rotation(axis_equations[0][0],axis_equations[0][1],m,b)

    return adjusted_axis, angle_rotation

def find_axis(image):

    #Apply Hough Transform (returns the vector of lines) into the canny edge detection image
    accumulator_threshold = 150 # Threshold to consider a line ("votes necessaries to be considered line")
    hough_lines = cv2.HoughLines(image, 1, np.pi / 180, accumulator_threshold)

    axis_lines = find_axis_lines(hough_lines)

    axis_lines_coordinates = translate_polar_coordinates(image, axis_lines)

    adjusted_axis, angle_rotation = find_perpendicular_lines(axis_lines_coordinates)

    return adjusted_axis, angle_rotation

def draw_axis_lines(src_image, lines,processed_image):
    for line in lines:
        cv2.line(processed_image, ( line['x1'],  line['y1']), ( line['x2'],  line['y2']), (0, 0, 0), 80)
        cv2.line(src_image, ( line['x1'],  line['y1']), ( line['x2'],  line['y2']), (255, 0, 0), 2)
        pass

def find_and_draw_axis(src_image):
    processed_image = process_image(src_image)
    axis, angle_rotation = find_axis(processed_image)
    draw_axis_lines(src_image, axis, processed_image)
    processed_image = remove_small_components(processed_image)

    return src_image, processed_image, angle_rotation
