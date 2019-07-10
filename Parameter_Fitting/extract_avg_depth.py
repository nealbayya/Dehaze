from os import listdir
import cv2 as cv
import numpy as np

PATH = "regression_data/depth/"

scenes = [s for s in listdir(PATH) if not s[0] in {'.', '_'}]
img2depth = {}

max_depth = 0
maxd_img = ""
min_depth = np.inf
mind_img = ""

for s in scenes:
    left_disp = cv.imread(PATH+s+"/disp1.png") / 3.0
    left_disp = np.clip(left_disp, a_min = np.min(left_disp+0.05), a_max = None)
    right_disp = cv.imread(PATH+s+"/disp5.png") / 3.0
    right_disp = np.clip(right_disp, a_min = np.min(right_disp+0.05), a_max = None)    

    left_depth = 3740*0.016 / left_disp
    right_depth = 3740*0.016 / right_disp
    img2depth[s+'_0'] = np.mean(left_depth)
    img2depth[s+'_1'] = np.mean(right_depth)

    if np.amax(left_depth) > max_depth:
        max_depth = np.amax(left_depth)
        maxd_img = s+'_0'
    elif np.amax(right_depth) > max_depth:
        max_depth = np.amax(right_depth)
        maxd_img = s+'_1'

    if np.amin(left_depth) < min_depth:
        min_depth = np.amin(left_depth)
        mind_img = s+'_0'
    elif np.amin(right_depth) < min_depth:
        min_depth = np.amin(right_depth)
        mind_img = s+'_1'

print("Image to Avg. Depth: {}".format(img2depth))
print("Max Depth: {}, Max Depth Scene: {}".format(max_depth, maxd_img))
print("Min Depth: {}, Min Depth Scene: {}".format(min_depth, mind_img))
    
