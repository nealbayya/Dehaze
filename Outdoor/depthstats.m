
depth_dir = 'data/depth/'; depth_list = dir(depth_dir);

image_names{1} = 'IMG_7033';
image_names{2} = 'IMG_7411';
image_names{3} = 'IMG_7460';
image_names{4} = 'IMG_8411';
image_names{5} = 'IMG_8416';
image_names{6} = 'IMG_8503';
image_names{7} = 'IMG_8509';
image_names{8} = 'IMG_8559';
image_names{9} = 'IMG_8583';
image_names{10} = 'IMG_8602';
image_names{11} = 'IMG_8612';
image_names{12} = 'IMG_8895';
image_names{13} = 'IMG_8905';

for j = 1: length(image_names)
depthname = [image_names{j}, '_depth.mat'];
depth_loc = [depth_dir,depthname];
load(depth_loc);
d = imDepth/1000;
d = nonzeros(d);
avgd = mean(d);
mind = min(d);
maxd = max(d);
ranged = maxd - mind;
fprintf([image_names{j}, '?', '%0.5f', '?', '%0.5f', '?', '%0.5f', '?', '%0.5f\n'], avgd, mind, maxd, ranged);
end