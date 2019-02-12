import numpy as np
import cv2 as cv
import random

dis = cv.imread('DIS/Cloth1?2.png', 0)/3.0
disref = cv.imread('2005Data/Cloth1/disp5.png', 0)/3.0
clearimg = cv.imread('2005Data/Cloth1/view5.png', 1)/255.0
for i in range(1, dis.shape[0]):
    for j in range(dis.shape[1]):
        if (dis[i,j] == 0):
            dis[i, j] = dis[i-1, j]
        if (disref[i,j] == 0):
            disref[i, j] = disref[i-1, j]
for i in range(dis.shape[0]):
    for j in range(1, dis.shape[1]):
        if (dis[i,j] == 0):
            dis[i, j] = dis[i, j-1]
        if (disref[i,j] == 0):
            disref[i, j] = disref[i, j-1]
dis = np.clip(dis, 1, dis.shape[1])
disref = np.clip(disref, 1, dis.shape[1])

depth = 3740*0.016/dis
depthref = 3740*0.016/disref

alpha = random.uniform(0.75, 1.0)
beta = random.uniform(0, 1.5)
print(alpha, beta)
t = np.exp(-beta*cv.merge((depthref, depthref, depthref)))
I = np.multiply(clearimg,t)+alpha*(1-t)
A = np.max(I)
B = beta+0.07
print(A, B)
T = np.exp(-B*cv.merge((depth, depth, depth)))
J = np.divide((I-A*(1-T)),T)
I*=255
J*=255
I = I.astype(np.uint8)
J = J.astype(np.uint8)
cv.imwrite('7Frame.png', I)
cv.imwrite('7Frame2.png', J)
