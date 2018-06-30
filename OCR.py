# -*-coding:utf-8-*-

import time

time1 = time.time()
import pytesseract
from PIL import Image

img1 = Image.open('E:\\document\\学信网\\show1.jpg')
code = pytesseract.image_to_string(img1, lang="chi_sim")
print(code)

time2 = time.time()
print('总耗时：' + str(time2 - time1))
