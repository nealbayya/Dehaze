% This is a demo script for fidelity metrics for dehazed images with
% respect to originals.
% It accompanies the paper: 
% HAZERD: an outdoor scene dataset and benchmark for single image dehazing
% IEEE Internation Conference on Image Processing, Sep 2017
% The paper and additional information on the project are available at:
% https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/
% If you use this code, please cite the above paper.
%
% Authors: 
% Yanfu Zhang: yzh185@ur.rochester.edu
% Li Ding: l.ding@rochester.edu
% Gaurav Sharma: gaurav.sharma@rochester.edu
% 
% Last update: May 2017

clc;clear;close all;

dehaze_img = 'data/dehaze/IMG_8612_200.jpg';
dehaze_esti = 'data/dehaze/IMG_8612_200.mat';
img = 'data/img/IMG_8612_RGB.jpg';
depth = 'data/depth/IMG_8612_depth.mat';
trans = 'data/dehaze/IMG_8612_trans_200.mat';
% Compute fidelity metrics for dehazed image
dehaze_result = dehaze_metrics(dehaze_img,dehaze_esti,img,depth,trans);
