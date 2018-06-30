# -*- coding: utf-8 -*-

from PIL import Image

import numpy as np

im = np.array(Image.open("6iys.jpg"))

print(im)
print(im.shape,im.dtype)