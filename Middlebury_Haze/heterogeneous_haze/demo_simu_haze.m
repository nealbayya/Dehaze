% This is a demo script for haze simulation with the dataset and parameters given by
% the paper: 
% HAZERD: an outdoor scene dataset and benchmark for single image dehazing
% IEEE Internation Conference on Image Processing, Sep 2017
% The paper and additional information on the project are available at:
% https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/
% If you use this code, please cite our paper.
%
% Set different values to variables (pert_perlin, airlight, and
% visual_range) to generate different hazy images
%
% The simulated images will be save at /data/simu/
%
% Authors: 
% Yanfu Zhang: yzh185@ur.rochester.edu
% Li Ding: l.ding@rochester.edu
% Gaurav Sharma: gaurav.sharma@rochester.edu
%
% Last update: May 2017

clear;clc;

img_dir = 'data/img/'; img_list = dir(img_dir);
depth_dir = 'data/depth/'; depth_list = dir(depth_dir);
save_dir = 'data/simu/';

addpath('./utils');

% set parameters
pert_perlin = 0;    % 1 for add perlin noise
airlight = 0.76*ones(3,1);  % airlight in the range [0,1]
visual_range = [0.05,0.1,0.2,0.5,1]; % visual range in km

%{
for j = 1:nitems = length(img_list)-2
    img_name= [img_dir,img_list(j+2).name];
    depth_name = [depth_dir,depth_list(j+2).name];
    hazy_img = hazy_simu(img_name,depth_name,save_dir,pert_perlin,airlight,visual_range);
end

%}

img_name = [img_dir,'IMG_adirondack_perfect0_RGB.png'];
depth_name = [depth_dir,'IMG_adirondack_perfect0_depth.mat'];
hazy_img = hazy_simu(img_name,depth_name,save_dir,pert_perlin,airlight,visual_range);
