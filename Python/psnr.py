import numpy as np
import cv2 as cv

img1 = cv.imread('6Frame2.png', 1)/255.
img2 = cv.imread('2005Data/Cloth3/view5.png', 1)/255.
img3 = cv.imread('6Frame.png', 1)/255.

def PSNR(y_true, y_pred):
    return 48.1308036087-10.*np.log(np.mean(np.square(255.*(y_pred - y_true)))) / np.log(10.)

def MSE(y_true, y_pred):
    return np.mean(np.square(255*(y_pred - y_true)))

def SSIM(y_true, y_pred):
    ux = np.mean(y_pred)
    uy = np.mean(y_true)
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (2*ux*uy+c1)*(2*np.std(y_pred)*np.std(y_true)+c2)
    denom =(ux**2+uy**2+c1)*(np.var(y_pred)+np.var(y_true)+c2)
    return ssim/np.clip(denom, 0.0000001, np.inf)

cv.imshow('Temp', img1)
cv.imshow('Temp2', img2)
cv.imshow('Temp3', img3)
cv.waitKey(0)
print(img1.shape)
print((MSE(img1, img2))**0.5)
print(PSNR(img1, img2))
print(SSIM(img1, img2))
print((MSE(img3, img2))**0.5)
print(PSNR(img3, img2))
print(SSIM(img3, img2))
