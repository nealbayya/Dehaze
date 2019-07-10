#Neal Bayya
#Generate crops for outdoor hazy scenes
from os import listdir
import glob
import random
import numpy as np
import numpy.matlib
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.io as sio

def generate_crops(model_input, size=128, npatches=10):
    [rows, columns, nchannels] = model_input.shape
    crops = []
    for i in range(npatches):
        ul_row = random.randint(0,rows-size)
        ul_col = random.randint(0,columns-size)
        patch_input = model_input[ul_row:ul_row+size, ul_col: ul_col+size,:]
        crops.append(patch_input)
    return np.array(crops)


def crops_master(images, img2params, scene2depth, cropsize, ncrops):
    print(img2params)
    modelio = open(path+'fourchannel_io.txt', 'w')
    for img_num, s in enumerate(images):
        beta = img2params[s][1]
        scene_name = s[0: s.rindex('_')] + '_RGB' + s[s.index('.'):]
        depth_map = scene2depth[scene_name]
        img = cv.imread(image_path + s)
        img = img.astype(np.float64)
        #DEBUGGING
        #test = Image.fromarray(img, 'RGB')
        #test.show()
        #print(img.shape)
        #print(depth_map.shape)
        #break
        #END DEBUGGING
        hazy_tensor = np.dstack((img, depth_map))
        tcrops = generate_crops(hazy_tensor, cropsize, ncrops)
        for c, tcrop in enumerate(tcrops):
            #Written format of each crop: IMG_CROP
            cn = s[0:s.rindex('.')] + "_" + str(c) + '.npz'
            modelio.write(cn + ' ' + str(beta) + '\n' )
            np.savez_compressed(crops_path+cn, hazyin = tcrop)
    modelio.close()

def extract_depth(scenes, scale = 1.0):
    scene2depth = {}
    for s in scenes:
        depth_name = s[0:s.rindex('_')] + '_depth.mat'
        depth = sio.loadmat(depth_path + depth_name)['imDepth']
        scene2depth[s] = depth
    return scene2depth

def main():
    global path, depth_path, image_path, crops_path 
    path = 'regression_data/HazeRD_data/'
    depth_path  = "regression_data/HazeRD_data/depth/"
    ground_truth_path = path + "img/"
    image_path = path + "simu/"
    crops_path = path + "crops4c/"
    ncrops = 5
    cropsize = 128
    
    #generate crops
    img2params = {}
    for i, l in enumerate(open(path+'scene_beta.txt', 'r')):
        split = l.split("?")
        img = split[0] 
        alpha = float(split[1])
        beta = float(split[2])
        img2params[img] = (alpha, beta)
    images = sorted([i for i in listdir(image_path) if not i.startswith(('.', '_'))])
    ground_truth = sorted([i for i in listdir(ground_truth_path) if not i.startswith(('.', '_'))])
    scene2depth = extract_depth(ground_truth, 1.0) #3.0 for Middlebury 2005 Small
    crops_master(images, img2params, scene2depth, cropsize, ncrops)
    
if __name__ == '__main__':
    main()
