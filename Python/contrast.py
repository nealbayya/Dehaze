import numpy as np
import cv2 as cv

img = cv.imread('rocks0.jpg', 1)
#img = cv.blur(input, (3, 3))

print(img.shape)
nimg = np.zeros((img.shape[0], img.shape[1], 3))
T = 20
a = 7
b = 7
h, w, d = img.shape
for i in range(h):
    for j in range(w):
        avg = np.mean(img[max(0, i-a):min(h, i+a), max(0, j-b):min(w, j+b)])
        temp = img[max(0, i-a):min(h, i+a), max(0, j-b):min(w, j+b)].copy()
        temp = temp.reshape(temp.shape[0]*temp.shape[1]*temp.shape[2])
        mx = np.mean(temp[np.argsort(temp)[-5:]])
        #print(mx)
        navg = avg-(mx-avg)
        nimg[i,j] = navg + 2*(img[i,j]-avg)
        #exit(0)
nimg = nimg.astype(np.uint8)
cv.imshow('Frame', nimg)
cv.imwrite('t0.jpg', nimg)
cv.waitKey(0)
