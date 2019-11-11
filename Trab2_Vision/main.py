from parabola import *
from axis import *

import cv2
import numpy as np
import math
import argparse

np.warnings.filterwarnings('ignore')


def detect_axis_and_parabola(image_name):
    image = cv2.imread(image_name)
    image, edge_map_image, angle_rotation = find_and_draw_axis(image)
    find_and_draw_parabola(image,edge_map_image, angle_rotation)
    cv2.imshow(image_name, image)


# main()
if __name__ == '__main__' :

    parser = argparse.ArgumentParser(description='Receive image')
    parser.add_argument('-i', '--img', required=True, type=str, help='Input image path')
    args = parser.parse_args()
    detect_axis_and_parabola(args.img)

    key = cv2.waitKey(0)
    if key == 27:  # wait for ESC
        print('Key ESC pressed.')
        cv2.destroyAllWindows()
