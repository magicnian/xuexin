import numpy as np

from PIL import Image

from keras.preprocessing import image

import cv2

import re

im = np.array(Image.open('3afchr.jpg'))

print(im.shape)

# im2 = image.img_to_array(image.load_img('3afchr.jpg', grayscale=True, target_size=(70, 160)))
im2 = image.img_to_array(Image.open('3afchr.jpg'))
print(im2.shape)

im3 = np.array(cv2.imread('3afchr.jpg', 0))

print(im3.shape)

s = '2\\abcd.jpg'

result = re.search('[a-z|A-Z|0-9]{4}.jpg',s)
print(result)
