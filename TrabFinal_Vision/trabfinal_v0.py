#trabalho final - Visão

# Author: Thiago Dal Moro

import numpy as np
import cv2

# referencias de cores para a bola amarela
# low_yellow = np.array([18, 94, 140])
# up_yellow = np.array([48, 255, 255])

video = cv2.VideoCapture("CTL.mp4")

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture("CTL.mp4")
        continue

    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # low_yellow = np.array([18, 94, 140])
    # up_yellow = np.array([48, 255, 255])
    # mask = cv2.inRange(hsv, low_yellow, up_yellow)
    # edges = cv2.Canny(mask, 75, 150)
    # edges = cv2.Canny(frame, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, None, 0, 0)
    # print(lines)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 5)

    cv2.imshow("frame", orig_frame)
    cv2.imshow("edges", edges)

    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()