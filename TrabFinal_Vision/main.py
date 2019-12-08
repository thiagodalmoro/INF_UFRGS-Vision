# from tkinter import X
# from tkinter.ttk import Style
#import PIL.Image, PIL.ImageTk
import imutils as imutils
from PIL import Image, ImageEnhance
# import tkinter as tk
import math
from collections import deque
# from imutils.video import VideoStream
import numpy as np
# import argparse
import cv2
from scipy import spatial
import argparse

NINETY_DEGREES = np.pi/2.0
angle_threshold = np.pi / 16
BALL_MIN_AREA = 33

def slope_intercept_equation(coordinates):
    x1 = coordinates['x1']
    x2 = coordinates['x2']
    y1 = coordinates['y1']
    y2 = coordinates['y2']

    m = (y2 - y1) / (x2 - x1)
    b = -(m * x2) + y2

    return m,b

def find_lines_equation(lines):
    axis_equations = []
    for line_coords in lines:
        m,b = slope_intercept_equation(line_coords)
        axis_equations.append([m,b])

    return axis_equations

def angle_between_lines(m1,b1,m2,b2):
    try:
        division = (m2 - m1) / (1 + m2*m1)
    except:
        return 0
    angle = np.arctan(division)
    return angle

def translate_polar_coordinates (image, axis_lines):
    # Calulate the image larger measure (lenght or width) to fit the lines we will draw in the image
    img_max_size = max(image.shape) * 1.1


    lines_list = []
    # We iterate over the axis_lines
    for line in axis_lines:
        #We want to transform polar coordinates to "normal" coordinates
        # for rho, theta in line:
        line = line[0]
        # print('a')
        # print(line)
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

def find_axis(image):

    #Apply Hough Transform (returns the vector of lines) into the canny edge detection image
    accumulator_threshold = 150 # Threshold to consider a line ("votes necessaries to be considered line")
    hough_lines = cv2.HoughLines(image, 1, np.pi / 180, accumulator_threshold)

    axis_lines_coordinates = translate_polar_coordinates(image, hough_lines[0:5])

    return axis_lines_coordinates

def find_rede(frame):
    mapa = process_image(frame)
    lines = find_axis(mapa)
    lines = lines[0:10]
    axis_equations = find_lines_equation(lines)
    i = 0
    for l in axis_equations:
        m1,b1 = l[0], l[1]
        m_horizontal = 0
        b_horizontal = 1
        angle1 = abs(angle_between_lines(m1,b1,m_horizontal,b_horizontal))
        angle1 = abs(angle1 - NINETY_DEGREES)
        if angle1 <= angle_threshold:
            rede = lines[i]
            break
        i+=1
    return rede, m1, b1

def find_ball(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key = cv2.contourArea)
        if cv2.contourArea(contour) > BALL_MIN_AREA:
            test = np.array(contour.reshape(-1,2))
            # two points which are fruthest apart will occur as vertices of the convex hull
            candidates = test[spatial.ConvexHull(test).vertices]

            # get distances between each pair of candidate points
            dist_mat = spatial.distance_matrix(candidates, candidates)

            # get indices of candidates that are furthest apart
            i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)


            p1, p2 = candidates[i], candidates[j]

            delta_x = abs((p1[0] - p2[0]) / 2)
            delta_y = abs((p1[1] - p2[1]) / 2)

            x = int(min(p1[0],p2[0]) + delta_x)
            y = int(min(p1[1],p2[1]) + delta_y)

            dist = int(math.sqrt((x - p1[0])**2 + (y - p1[1])**2))

            return (x,y), dist
    return (0,0), 0


H_MIN = 0;
H_MAX = 256;
S_MIN = 0;
S_MAX = 256;
V_MIN = 0;
V_MAX = 256;
# default capture width and height
FRAME_WIDTH = 640;
FRAME_HEIGHT = 480;
# max number of objects to be detected in frame
MAX_NUM_OBJECTS=50;
# minimum and maximum object area
MIN_OBJECT_AREA = 20*20;
MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;


#
# def switches(switch):
#     cv2.createTrackbar(switch, 'Video Frame', 0, 1, applyVal)
#     cv2.createTrackbar('lower', 'Video Frame', 0, 255, applyVal)
#     cv2.createTrackbar('upper', 'Video Frame', 0, 255, applyVal)
#     cv2.createTrackbar('Blur', 'Video Frame', 3, 5, applyVal)


#
# # determine upper and lower HSV limits for (my) skin tones



def main(debug_mode):
    rede = []
    camera = cv2.VideoCapture("video.mp4")
    # camera = cv2.VideoCapture(0)
    cv2.namedWindow('Original Output')
    balls_positions = deque()
    toque_recente = 0
    has_to_write = 0
    #
    # switch = '0 : OFF \n1 : ON'
    # switches(switch)
    i = 0
    while (True):
        ret, frame = camera.read()
        if not ret:
            continue
        #
        # lower_i = cv2.getTrackbarPos('lower', 'Original Output')
        # upper_i = cv2.getTrackbarPos('upper', 'Original Output')
        # s = cv2.getTrackbarPos(switch, 'Original Output')

        # referencias de cores para a bola verde
        # low_green = np.array([35, 40, 19])
        # up_green = np.array([82, 246, 139])

        lower = np.array([20, 207, 139], dtype="uint8")
        upper = np.array([83,255,255], dtype="uint8")
        # switches(switch)

        # lower = np.array([35, 40, 19], dtype="uint8")
        # upper = np.array([82, 246, 139], dtype="uint8")
        #


        # # switch to HSV

        #green_low_new
        # lower = np.array([17, 34, 18], dtype="uint8")
        # upper = np.array([53, 139, 102], dtype="uint8")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        # #bola azul
        # lower = np.array([55, 80, 26], dtype="uint8")
        # upper = np.array([110, 255, 187], dtype="uint8")

        # find mask of pixels within HSV range
        mask = cv2.inRange(hsv, lower, upper)

        # cv2.imshow("Original", frame)
        # # cv2.imshow("HSV", hsv)
        # cv2.imshow("skinMask", mask)

        # atualizar para a skinmask

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        # mask = cv2.inRange(hsv, greenLower, greenUpper)

        #
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)
        #
        # cv2.imshow("skinMask2", mask)

        #
        #
        # # find contours in the mask and initialize the current
        # # (x, y) center of the ball
        # cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        #                                    cv2.CHAIN_APPROX_SIMPLE)


        kernel = np.ones((5,5), np.uint8)
        dilated_image = cv2.dilate(mask, kernel, iterations=1)

        #Desenha a rede
        if not rede:
            rede, m, b = find_rede(frame)
            y_top = 0
            x_top = int((y_top - b) / m)
            y_bottom = frame.shape[0]
            x_bottom = int((y_bottom - b) / m)
            left_limit_rede = min(x_top,x_bottom)
            right_limit_rede = max(x_top,x_bottom)

        cv2.line(frame,(rede['x1'],rede['y1']),(rede['x2'],rede['y2']),(0,255,0),2)
        if right_limit_rede == x_top:
            cv2.circle(frame, (x_top,y_top), 10, (255, 0, 0), 2)
            cv2.circle(frame, (x_bottom,y_bottom), 10, (0, 0, 255), 2)
        else:
            cv2.circle(frame, (x_top,y_top), 10, (0, 0, 255), 2)
            cv2.circle(frame, (x_bottom,y_bottom), 10, (255, 0, 0), 2)

        #Desenha a bola
        ball_coord, ball_radius = find_ball(dilated_image)
        if ball_radius:
            if ball_radius > 5 or len(balls_positions) > 0:
                cv2.circle(frame, ball_coord, ball_radius, (0, 255, 0), 2)
                balls_positions.append(ball_coord)
                if len(balls_positions) > 4:
                    balls_positions.popleft()
                    if debug_mode:
                        cv2.circle(frame, balls_positions[2], ball_radius, (255, 0, 0), 2)
                        cv2.circle(frame, balls_positions[1], ball_radius, (0, 0, 255), 2)
                        cv2.circle(frame, balls_positions[0], ball_radius, (0, 255, 255), 2)

                    y_atual = balls_positions[3][1]
                    y_passado = balls_positions[2][1]
                    y_retrasado = balls_positions[1][1]
                    y_reretrasado = balls_positions[0][1]

                    if toque_recente:
                        toque_recente -=1
                        #Tinha que estar descendo a bola (y estava aumentando) e de repente subir (y diminuir) com uma margem de 3 pixels
                    elif (y_atual + 3 < y_passado and y_passado > y_retrasado) or (y_atual + 3 < y_retrasado and y_retrasado > y_reretrasado):
                        i+=1
                        print('b-3 ', balls_positions[0])
                        print('b-2 ', balls_positions[1])
                        print('b-1 ', balls_positions[2])
                        print('b0 ', balls_positions[3])
                        print(i, ' TOCOU')
                        toque_recente = 3
                        has_to_write = 5
                    else:
                        min_old_y = min(y_passado, y_retrasado, y_reretrasado) - 2
                        max_old_y = max(y_passado, y_retrasado, y_reretrasado) + 2
                        #se considerar as posicoes passadas da bola e estarem no mesmo y (com ate 4 pixels de margem) pode ser que bola estivesse parada
                        if max_old_y - min_old_y <=0:
                            x_atual = balls_positions[3][0]
                            x_passado = balls_positions[2][0]
                            x_retrasado = balls_positions[1][0]
                            x_reretrasado = balls_positions[0][0]
                            min_old_x = min(x_passado, x_retrasado, x_reretrasado)
                            max_old_x = max(x_passado, x_retrasado, x_reretrasado)
                            #Testa se a bola se mexeu mais de 5 em x, o que significaria que houve toque de fato pra frente/tras
                            if x_atual < min_old_x:
                                if x_atual < min_old_x - 5:
                                    print('b-3 ', balls_positions[0])
                                    print('b-2 ', balls_positions[1])
                                    print('b-1 ', balls_positions[2])
                                    print('b0 ', balls_positions[3])
                                    print(i, ' TOCOU')
                                    toque_recente = 3
                                    has_to_write = 5
                            #Testa se a bola se mexeu mais de 5 em x, o que significaria que houve toque de fato pra frente/tras
                            if x_atual > max_old_x:
                                if x_atual > max_old_x + 5:
                                    print('b-3 ', balls_positions[0])
                                    print('b-2 ', balls_positions[1])
                                    print('b-1 ', balls_positions[2])
                                    print('b0 ', balls_positions[3])
                                    print(i, ' TOCOU')
                                    toque_recente = 3
                                    has_to_write = 5


        if has_to_write:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (0, 255, 255)
            thickness = 2
            cv2.putText(frame, 'Tocou!', org, font, fontScale, color, thickness, cv2.LINE_AA)
            has_to_write -=1

        cv2.imshow("Original", frame)
        # cv2.imshow("HSV", hsv)
        # cv2.imshow("skinMask", dilated_image)


        if cv2.waitKey(100) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


# def applyVal(value):
#     print('Applying blur!', value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process flags')
    parser.add_argument('-d', '--debug', default=0,
                        type=int, help='Debug Mode')
    args = parser.parse_args()
    main(args.debug)
