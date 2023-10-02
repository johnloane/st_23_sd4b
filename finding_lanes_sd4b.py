import cv2
import numpy as np

image = cv2.imread("road.jpg")
lane_image = np.copy(image)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
print(gray)
cv2.imshow("Gray", gray)
cv2.waitKey(0)