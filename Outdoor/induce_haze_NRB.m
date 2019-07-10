%Neal Bayya
%September 14, 2018
%TJHSST, Senior Reseach Lab


%Objective: Test the hazeRD inducing model


clear;clc;

img_dir = 'data/img/'; img_list = dir(img_dir);
depth_dir = 'data/depth/'; depth_list = dir(depth_dir);
save_dir = 'data/simu/perlin/';

addpath('./utils');

% set parameters
pert_perlin = 0;    % 1 for add perlin noise
airlight = 0.76*ones(3,1);  % airlight in the range [0,1]
visual_range = [0.05,0.1,0.2,0.5,1]; % visual range in km

nimages = length(img_list)-2;

%{
for j = 1:length(img_list)-2
    img_name= [img_dir,img_list(j+2).name];
    depth_name = [depth_dir,depth_list(j+2).name];
    hazy_img = hazy_simu(img_name,depth_name,save_dir,pert_perlin,airlight,visual_range);
end
%}

pert_perlin = 1;
visual_range = [0.075]; %NRB, should see 750 marker in simu file
img_name= [img_dir,img_list(1+2).name];
disp(img_name);
depth_name = [depth_dir,depth_list(1+2).name];
hazy_img = hazy_simu(img_name,depth_name,save_dir,pert_perlin,airlight,visual_range);
