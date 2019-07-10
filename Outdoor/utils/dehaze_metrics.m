function dehaze_result = dehaze_metrics(dehaze_img,dehaze_esti,img,depth,trans)
% This is the function for haze simulation described in the paper: 
% HAZERD: an outdoor scene dataset and benchmark for single image dehazing
% IEEE International Conference on Image Processing, Sep 2017
% The paper and additional information on the project are available at:
% https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/
% If you use this code, please cite our paper.
%
% Input:
%   dehaze_img: the directory and name of a dehazed image, the name should be
%               ..._{visual range value}.jpg
%   dehaze_esti: the corresponding directory and name of the estimated transmission, in
%                .mat file, the name should be in the format of ..._{visual range
%                value}.mat, the data should be saved as the name 'trans_esti'
%   img: the directory and name of the haze-free RGB image, the name 
%        should be in the format of ..._RGB.jpg 
%   depth: the corresponding directory and name of the depth map, in
%          .mat file, the name should be in the format of ..._depth.mat
%
% Output:
%   dehaze_result: a struct with field 'RMS', 'SSIM' and 'CIEDE2000'
%
% Authors: 
%   Yanfu Zhang: yzh185@ur.rochester.edu
%   Li Ding: l.ding@rochester.edu
%   Gaurav Sharma: gaurav.sharma@rochester.edu
% 
% Last update: May 2017

fprintf('Evaluating dehazed image: %s\n',dehaze_img);
I = imread(img);
I0 = imread(dehaze_img);
load(depth);load(dehaze_esti);load(trans);

% evaluate RMS error in transmission. Compute only over valid depth values.
% Invalid depth values are assumed to be set to 0
rms_trans = rms(transmission(imDepth>0)-trans_esti(imDepth>0));
fprintf('Transmission RMS: %f\n',rms_trans);
% evaluate SSIM between original and dehazed RGB images
ssim_img = ssim(I,I0);
fprintf('SSIM: %f\n',ssim_img);
% evaluate CIEDE2000 between original and dehazed RGB images
[h,w,c] = size(I);
ciede2000_img = mean(mean(deltaE2000(reshape(rgb2lab(I),h*w,c),reshape(rgb2lab(I0),h*w,c))));
fprintf('CIEDE2000: %f\n',ciede2000_img);

dehaze_result = struct('RMS',rms_trans,'SSIM',ssim_img,'CIEDE2000',ciede2000_img);

end