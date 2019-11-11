from parabola import *
from axis import *

import cv2
import numpy as np
import math

np.warnings.filterwarnings('ignore')


def detect_axis_and_parabola(image_name):
    image = cv2.imread(image_name)
    image, edge_map_image, angle_rotation = find_and_draw_axis(image)
    find_and_draw_parabola(image,edge_map_image, angle_rotation)
    cv2.imshow(image_name, image)


# main()
if __name__ == '__main__' :

    detect_axis_and_parabola('exemplo1.jpg')
    detect_axis_and_parabola('exemplo2.jpg')
    detect_axis_and_parabola('exemplo3.jpg')

    key = cv2.waitKey(0)
    if key == 27:  # wait for ESC
        print('Key ESC pressed.')
        cv2.destroyAllWindows()
