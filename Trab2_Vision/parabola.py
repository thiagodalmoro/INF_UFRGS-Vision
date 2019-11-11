import numpy as np
import cv2

np.warnings.filterwarnings('ignore')


def draw_parabola(image,equation, angle_rotation):
    for i in range(-1000,1000,1):
        x0 = i
        y0 = (equation[0] * (x0**2)) + (equation[1] * x0) + equation[2]
        x = x0 * np.cos(-angle_rotation) - y0 * np.sin(-angle_rotation)
        y = y0 * np.cos(-angle_rotation) + x0 * np.sin(-angle_rotation)

        draw_point(image,int(x),int(y))

def draw_point(image,x,y):
    center = (x,int(y))
    cv2.circle(image,center, 1, (0,0,255), 2)

def find_parabola_points(edge_map_image, angle_rotation,image):
    width = len(edge_map_image[0])
    height = len(edge_map_image)

    coordinates = []
    for i in range(width):
        for j in range(height):
            if edge_map_image[j][i] != 0:
                x0 = i
                y0 = j
                x = x0 * np.cos(angle_rotation) - y0 * np.sin(angle_rotation)
                y = y0 * np.cos(angle_rotation) + x0 * np.sin(angle_rotation)

                coordinates.append([x,y])

    return coordinates

def find_parabola_equation(points):
    coefficients = []
    values = []

    #For each pair of points we set the equation (axˆ2, bx, c = y)
    for x,y in points:
        coefficients.append([x**2, x, 1])
        values.append(y)

    #We create the A and y from Ax = y
    A = np.array(coefficients)
    y = np.array(values)

    #We solve Ax = y by least squares
    equation = np.linalg.lstsq(A,y)[0]

    return equation

def find_and_draw_parabola(image, edge_map_image,angle_rotation):
    parabola_points = find_parabola_points(edge_map_image, angle_rotation,image)
    parabola_equation = find_parabola_equation(parabola_points)
    draw_parabola(image, parabola_equation, angle_rotation)
