import cv2
import numpy as np

image = cv2.imread(r'E:\document\test.jpg')

# cv2.imshow("Original",image)

cv2.waitKey(0)

# 高斯滤波
blurred = np.hstack([cv2.GaussianBlur(image, (11, 11), 0)
                     ])

cv2.imwrite(r'E:\document\gaussianBlur.jpg', blurred)

cv2.imshow("Averaged", blurred)
cv2.waitKey(0)

# res = cv2.resize(blurred, (52, 20), interpolation=cv2.INTER_CUBIC)

# cv2.imwrite(r'E:\document\gaussianBlur_resize.jpg', res)
# cv2.imshow('res', res)
# cv2.waitKey(0)
