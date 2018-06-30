import cv2

img = cv2.imread(r'E:\document\show1.jpg')
# print(img)

# cv2.imshow('ori',img)

res = cv2.resize(img, (800, 320), interpolation=cv2.INTER_CUBIC)

cv2.imwrite(r'E:\document\test.jpg', res)

# cv2.imshow('ori',img)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
