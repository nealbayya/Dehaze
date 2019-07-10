clear;
clc;

rng(0,'twister');
vr_min = 0.05;
vr_max = 1.0;

img_dir = 'data/img/'; img_list = dir(img_dir);
depth_dir = 'data/depth/'; depth_list = dir(depth_dir);
save_dir = 'data/simu/';

addpath('./utils');

% set parameters
pert_perlin = 0;    % 1 for add perlin noise
airlight_param = 0.76;
airlight = airlight_param*ones(3,1);  % airlight in the range [0,1]
%visual_range = [0.05,0.1,0.2,0.5,1]; % visual range in k

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
%image_names{14} = 'IMG_9562';
%image_names{15} = 'IMG_9564';

run_num = length(image_names); %

scene2beta = fopen('scene_beta.txt', 'wt');

for j = 1:length(image_names)
    vr = (vr_max-vr_min).*rand(30,1) + vr_min;
    
    img_name = [image_names{j}, '_RGB.jpg'];
    img_loc= [img_dir,img_name];
    
    depthname = [image_names{j}, '_depth.mat'];
    depth_loc = [depth_dir,depthname];
    
    [fnames, beta] = hazy_simu_regression(img_loc,depth_loc,save_dir,pert_perlin,airlight,vr);
    for k = 1: length(vr)
        fprintf(scene2beta, [fnames{k}, '?', '%0.5f', '?', '%0.5f\n'], airlight_param, beta(k));
    end
end

fclose(scene2beta);
