#trabalho final - Visão

# Author: Thiago Dal Moro

import cv2
import numpy as np
import time


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


def find_line_intersection(lines):
    line1_point1 = [lines[0]['x1'], lines[0]['y1']]
    line1_point2 = [lines[0]['x2'], lines[0]['y2']]
    line2_point1 = [lines[1]['x1'], lines[1]['y1']]
    line2_point2 = [lines[1]['x2'], lines[1]['y2']]
    x_intersec,y_intersec = calculate_line_intersection((line1_point1, line1_point2), (line2_point1, line2_point2))

    return x_intersec,y_intersec

def draw_point(image,x,y):
    center = (x,int(y))
    cv2.circle(image,center, 1, (0,0,255), 2)

def draw_ball(frame,x,y):
    # draw_ball green


# # referencias de cores para a bola verde
#     low_red = np.array([20, 207, 139])
#     up_red = np.array([83, 255, 255])


def rescale (frame, percent):

    frame_width = int(frame.shape[1] * percent/ 100)
    frame_height = int(frame.shape[0] * percent/ 100)
    tamanho = (frame_width, frame_height)
    frame_rescale = cv2.resize(frame, tamanho, interpolation=cv2.INTER_AREA)
    return frame_rescale


def print_line(frame):
    start_pointx = (752, 21)
    end_pointx = (720, 679)
    # color = BLUE (B, G, R)
    color = (255, 0, 0)
    # pxl px
    thinckness = 3
    frame = cv2.line(frame, start_pointx, end_pointx, color, thinckness)
    frame_rescale = rescale(frame, 50)
    cv2.imshow('Img Rescale', frame_rescale)


def main():

    # allow the camera or video file to warm up
    time.sleep(2.0)

    cap = cv2.VideoCapture("CTL.mp4")

    while True:
        ret, frame = cap.read()

        print_line(frame)


        #draw_ball

        # referencias de cores para a bola verde
        low_red = np.array([20, 207, 139])
        up_red = np.array([83, 255, 255])

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


