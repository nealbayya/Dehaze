clear;
clc;

img_dir = 'data/midd_img/'; img_list = dir(img_dir);
depth_dir = 'data/midd_depth/'; depth_list = dir(depth_dir);
save_dir = 'data/midd_simu_betanoise/';

addpath('./utils');

% Are you going to use the training or test set?
imgset = 'training';
%imgset = 'test';

% Specify which resolution you are using for the stereo image set (F, H, or Q?)
imgsize = 'Q';
%imgsize = 'H';
%imgsize = 'F';


% set parameters
pert_perlin = 0;    % 1 for add perlin noise
airlight = 0.76*ones(3,1);  % airlight in the range [0,1]
visual_range = [0.05,0.1,0.2,0.5,1]; % visual range in km

image_names{1} = 'Adirondack';
image_names{2} = 'ArtL';
image_names{3} = 'Jadeplant';
image_names{4} = 'Motorcycle';
image_names{5} = 'MotorcycleE';
image_names{6} = 'Piano';
image_names{7} = 'PianoL';
image_names{8} = 'Pipes';
image_names{9} = 'Playroom';
image_names{10} = 'Playtable';
image_names{11} = 'PlaytableP';
image_names{12} = 'Recycle';
image_names{13} = 'Shelves';
image_names{14} = 'Teddy';
image_names{15} = 'Vintage';

run_num = length(image_names); %

for j = 1:run_num
    img0_name = ['IMG_', strcat(imgset,imgsize), '_', image_names{j}, '0_RGB.png'];
    img0_loc= [img_dir,img0_name];
    img1_name = ['IMG_', strcat(imgset,imgsize), '_', image_names{j}, '1_RGB.png'];
    img1_loc= [img_dir,img1_name];
    
    depth0name = ['IMG_', strcat(imgset,imgsize), '_', image_names{j}, '0_depth.mat'];
    depth0_loc = [depth_dir,depth0name];
    
    depth1name = ['IMG_', strcat(imgset,imgsize), '_', image_names{j}, '1_depth.mat'];
    depth1_loc = [depth_dir,depth1name];
    
    
    hazy_img0 = hazy_simu_betanoise(img0_loc,depth0_loc,save_dir,pert_perlin,airlight,visual_range);
    hazy_img1 = hazy_simu_betanoise(img1_loc,depth1_loc,save_dir,pert_perlin,airlight,visual_range);
end


