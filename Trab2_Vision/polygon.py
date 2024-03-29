# Python code to detect an arrow (seven-sided shape) from an image.
import numpy as np
import cv2



line = []

def mouse_drawing(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left click")
        print(line)

# Draw a diagonal blue line with thickness of 5 px

# Reading image
img2 = cv2.imread('exemplo1.jpg', cv2.IMREAD_COLOR)

# Reading same image in another variable and
# converting to gray scale.
img = cv2.imread('exemplo1.jpg', cv2.IMREAD_GRAYSCALE)

# Converting image to a binary image
# (black and white only image).
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

# Detecting shapes in image by selecting region
# with same colors or intensity.
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
	area = cv2.contourArea(contour)
	# print(area)

	# Shortlisting the regions based on there area.
	if area > 400:
		approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
		print(approx)

		if (len(approx)== 100):

			cv2.drawContours(img2, [approx], 0, (0, 0, 255), 2)


# Searching through every region selected to
# find the required polygon.
# for cnt in contours :
# 	area = cv2.contourArea(cnt)
#
# 	# Shortlisting the regions based on there area.
# 	if area > 400:
# 		approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
#
# 		# Checking if the no. of sides of the selected region is 7.
# 		if(len(approx) == 100):
# 			cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)

# Showing the image along with outlined arrow.
cv2.imshow('img2', img2)
cv2.imshow('img', img)
cv2.imshow('th', threshold)

# Exiting the window if 'q' is pressed on the keyboard.
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
