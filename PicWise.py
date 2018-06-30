import cv2
import numpy as np

image = cv2.imread(r'E:\document\test.jpg')

# cv2.imshow("Original",image)

cv2.waitKey(0)

# 领域均值滤波
blurred = np.hstack([cv2.blur(image, (30, 30))
                     ])

cv2.imwrite(r'E:\document\blur.jpg', blurred)

cv2.imshow("Averaged", blurred)
cv2.waitKey(0)
