from random import shuffle
from os import listdir
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import stats
from RGB_conversion import *
#_SRGB_8BIT_TO_LINEAR = [srgb_to_linear(i / 255.0) for i in range(256)]


all_hazy_scenes = sorted(listdir('simu/'))
for idx in range(13):
	hazy_scenes = all_hazy_scenes[idx*30: (idx+1)*30]
	#print(hazy_scenes)
	name2range = {}
	name2beta = {}

	for i, hazy_name in enumerate(hazy_scenes):
		#print("Processing {} of {}".format(i, len(hazy_scenes)))
		hazy = cv.imread('simu/' + hazy_name)
	
		hazy_shape = hazy.shape
		rows = hazy_shape[0]
		cols = hazy_shape[1]
		channels = hazy_shape[2]
		'''
		hazy_lrgb = np.zeros(hazy_shape)
		for r in range(rows):
			for c in range(cols):
				for z in range(channels):
					hazy_lrgb[r,c,z] = _SRGB_8BIT_TO_LINEAR[hazy[r,c,z]]
		'''
		l = np.sum(hazy, axis=-1)
		r = np.amax(l) - np.amin(l)
		name2range[hazy_name] = r

	beta_file = open("scene_beta.txt", "r")
	for l in beta_file:
		sp = l.split('?')
		beta = float(sp[-1])
		fname = sp[0]
		name2beta[fname] = beta
	
	range_par = []
	beta_par = []
	for name in hazy_scenes:
		range_par.append(name2range[name])
		beta_par.append(name2beta[name])

	slope, intercept, r_value, p_value, std_err = stats.linregress(range_par,beta_par)


	test_scene = hazy_scenes[0]
	output = test_scene[0: test_scene.rindex('_')] + "?" + str(slope) + "?" + str(intercept)
	print(output)

	#plt.scatter(range_par, beta_par, marker='o')
	#plt.show()



