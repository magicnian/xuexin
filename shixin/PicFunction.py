import cv2
import os

dirpath = 'E:\\document\\shixin\\gen\\1'
dstdir = 'E:\\document\\shixin\\gen\\dst\\1'

files = os.listdir(dirpath)

for name in files:
    im = cv2.imread(os.path.join(dirpath, name))
    # im = cv2.imread('E:\\document\\shixin\\zhixing\\zhixing\\2ape.png')
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(dstdir, name), thresh)
# cv2.imshow('t', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
