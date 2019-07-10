function save_name = hazy_simu_betanoise(img_name,depth_name,save_dir,varargin)
%Haze simulatory with noise added to beta coefficient
%   Neal Bayya, 2018

% Input:
%   img_name: the directory and name of a haze-free RGB image, the name 
%             should be in the format of ..._RGB.jpg 
%   depth_name: the corresponding directory and name of the depth map, in
%               .mat file, the name should be in the format of ..._depth.mat
%   save_dir: the directory to save the simulated images
%   varargin: optional
%   varargin{1}: 1 for adding perlin noise, default 0
%   varargin{2}: airlight, 3*1 matrix in the range [0,1]
%   varargin{3}: visual range, a vector of any size 
%
% Output:
%   save_name: image name of hazy image
%
% IMPORTANT NOTE: The code uses the convention that pixel locations with a
% depth value of 0 correspond to objects that are very far and for the
% simulation of haze these are placed a distance of 2 times the visual
% range.
%
% Authors: 
%   Yanfu Zhang: yzh185@ur.rochester.edu
%   Li Ding: l.ding@rochester.edu
%   Gaurav Sharma: gaurav.sharma@rochester.edu
% 
% Last update: May 2017

% parse inputs and set default values
% Set default parameter values. Some of these are used only if they are not
% passed in
pert_perlin = 0;
airlight=0.76;

A = zeros(1,1,3);
A(1,1,1) = airlight;A(1,1,2) = airlight;A(1,1,3) = airlight;
visual_range = [0.05,0.1,0.2,0.5,1]; % km
if ~isempty(varargin)
    pert_perlin = varargin{1};
end
if length(varargin) > 1
    A(1,1,1) = varargin{2}(1);A(1,1,2) = varargin{2}(2);A(1,1,3) = varargin{2}(3);
end
if length(varargin) > 2
    visual_range = varargin{3};
end

if exist(save_dir,'dir') ~= 7
    mkdir(save_dir);        % make dir if not exists
end

fprintf('Simulating hazy image for: %s\n',img_name);
for VR = visual_range
    fprintf('    Viusal range: %.3f km\n',VR);
    I0 = im2double(imread(img_name));
    [sx, sy, ~] = size(I0);
    %I0 = imread(img_name)/ 255.0;
    % convert sRGB to linear RGB
    I = srgb2lrgb(I0);
    
    load(depth_name);   
    d = imDepth/1000;   % convert meter to kilometer
    
    %compute beta param from average depth
    average_depth = mean(d(:));
    %beta_param = 0.03391/average_depth;
    beta_param = 0.0231;
    
%   Set regions where depth value is set to 0 to indicate no valid depth to
%   a distance of two times the visual range. These regions typically
%   correspond to sky areas
  
    %d(d==0) = 2*VR;
    
    %Modification by NRB 10/14
    
    d(d==0) = average_depth;
    
    % End modification
    	
%	Add perlin noise for visual vividness(see reference 16 in the HazeRD paper). The noise is added to the depth 
%	map, which is equivalent to change the optic distance. 
    if pert_perlin
        d = d.*((perlin_noise(zeros(size(d)))-0.5)+1);
    end
    
    % convert depth map to transmission
    %beta = [beta_param/VR,beta_param/VR,beta_param/VR];beta = reshape(beta,[1,1,3]);
    beta = repmat(beta_param/VR, sx, sy, 1);
    [beta, lframe, lf_superpos] = betanoise(beta);
    transmission = exp(bsxfun(@times,-beta,d));
    
    % Obtain simulated linear RGB hazy image. Eq. 3 in the HazeRD paper
    Ic = bsxfun(@times,transmission,I)+bsxfun(@times,1-transmission,A);
    
    % convert linear RGB to sRGB
    I2 = lrgb2srgb(Ic);
    
    % save result
    [foo,name,ext] = fileparts(img_name); 
    save_name = strrep(name,'_RGB',['_',num2str(VR*1000),ext]);
    imwrite(I2,[save_dir,save_name]);
    [bar,name,ext] = fileparts(depth_name); 
    save_name_trans = strrep(name,'_depth',['_trans_',num2str(VR*1000),ext]);
    save([save_dir,save_name_trans],'transmission');
    save_name_beta = strrep(name,'_depth',['_betanoise_',num2str(VR*1000),ext]);
    save([save_dir,save_name_beta],'beta', 'lframe', 'lf_superpos');
end
fprintf('\n');

end

function I = srgb2lrgb(I0)
    I = ((I0+0.055)/1.055).^(2.4);  
    I(I0<=0.04045) = I0(I0<=0.04045)/12.92;
end

function I2 = lrgb2srgb(I1)
    I2 = zeros(size(I1));
    for k = 1:3
        temp = I1(:,:,k);
        I2(:,:,k) = 12.92*temp.*(temp<=0.0031308)+(1.055*temp.^(1/2.4)-0.055).*(temp>0.0031308);
    end
end



