import numpy as np
import cv2 as cv
import glob
import random

folders = glob.glob("2005Data/*")
for folder in folders:
    print(folder.split('/')[1])
    limg = cv.imread(folder+'/view1.png', 1)/255.0
    ldis = cv.imread(folder+'/disp1.png', 0)/3
    rimg = cv.imread(folder+'/view5.png', 1)/255.0
    rdis = cv.imread(folder+'/disp5.png', 0)/3
    for i in range(1, ldis.shape[0]):
        for j in range(ldis.shape[1]):
            if (ldis[i,j] == 0):
                ldis[i, j] = ldis[i-1, j]
            if (rdis[i,j] == 0):
                rdis[i, j] = rdis[i-1, j]
    for i in range(ldis.shape[0]):
        for j in range(1, ldis.shape[1]):
            if (ldis[i,j] == 0):
                ldis[i, j] = ldis[i, j-1]
            if (rdis[i,j] == 0):
                rdis[i, j] = rdis[i, j-1]
    ldis = np.clip(ldis, 1, ldis.shape[1])
    rdis = np.clip(rdis, 1, rdis.shape[1])
    ldepth = 3740*0.016/ldis
    rdepth = 3740*0.016/rdis
    for c in range(10):
        alpha = random.uniform(0.75, 1.0)
        beta = random.uniform(0, 1.5)
        lt = np.exp(-beta*cv.merge((ldepth, ldepth, ldepth)))
        rt = np.exp(-beta*cv.merge((rdepth, rdepth, rdepth)))
        lI = 255*(np.multiply(limg,lt)+alpha*(1-lt))
        rI = 255*(np.multiply(rimg,rt)+alpha*(1-rt))
        lI = lI.astype(np.uint8)
        rI = rI.astype(np.uint8)
        name = folder.split('/')[1]
        print(name+'?'+str(c)+'?'+str(alpha)+'?'+str(beta))
        cv.imwrite('HazyImages/'+name+'?'+str(c)+'?0.png', lI)
        cv.imwrite('HazyImages/'+name+'?'+str(c)+'?1.png', rI)
