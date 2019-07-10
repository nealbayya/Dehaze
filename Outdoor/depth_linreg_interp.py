import matplotlib.pyplot as plt

depth_file = open("scene_depth.txt", "r")
#depth file has format of file?avgd?mind?maxd?ranged
linstats_file = open("scene_linreg.txt", "r")
#linstats file has format of file?slope?intercept

depth_selection = 'avgd'
stats_selection = 'slope'

scene2dstat = {}
scene2lstat = {}

for dm in depth_file:
	vec = dm.split("?")
	scene_name = vec[0]
	idx = -1
	if depth_selection == 'avgd': idx = 1
	elif depth_selection == 'mind': idx = 2
	elif depth_selection == 'maxd': idx = 3
	elif depth_selection == 'ranged': idx = 4
	dstat = vec[idx]
	scene2dstat[scene_name] = float(dstat)

for lsm in linstats_file:
	vec = lsm.split("?")
	scene_name = vec[0]
	idx = -1
	if stats_selection == "slope": idx = 1
	else: idx = 2
	lstat = vec[idx]
	scene2lstat[scene_name] = float(lstat)

ds_par = []
ls_par = []

for scene in scene2dstat.keys():
	ds_par.append(scene2dstat[scene])
	ls_par.append(scene2lstat[scene])

plt.scatter(ds_par, ls_par, marker='o')
plt.show()
