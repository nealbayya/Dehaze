import numpy as np
import cv2 as cv
"""
li = cv.imread('lampshade0.jpg')
ri = cv.imread('lampshade1.jpg')
cv.imshow('Left', li)
cv.imshow('Right', ri)
cv.waitKey(0)
"""
img = cv.imread('plastic1.png')
img = cv.Canny(img, 0, 50)
kernel = np.ones((3,3),np.uint8)
img = cv.dilate(img,kernel,iterations = 1)
cv.imshow('Frame', img)
cv.imwrite('plasticE.png', img)
cv.waitKey(0)
