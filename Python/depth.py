import numpy as np
import cv2 as cv

def ssd(a, b):
    return np.sum((a-b)**2)

def ncc(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

# dispairty map size (l-cw, w-cw)
def disparity(a, b, size):
    assert size%2 != 0
    c = size//2
    f = 60
    l, w = a.shape[:2]
    d = np.zeros((l, w))
    for i in range(c, l-c):
        for j in range(c, w-c):
            print(i, j)
            best = 0.6 #threshold
            y = -1
            for k in range(max(j-f, c), min(j+f, w-c)):
                match = ncc(a[i-c:i+c,j-c:j+c], b[i-c:i+c,k-c:k+c])
                if match > best:
                    best = match
                    y = k
            d[i, j] = 0 if (y==-1) else abs(y - j)/w*15
    return d

left = cv.imread('cone0.jpg', 1)
right = cv.imread('cone1.jpg', 1)

map = disparity(left, right, 5)
cv.imshow('Dispairty Map', map)
cv.waitKey(0)
