function im = perlin_noise(im,varargin)
% This is the function for adding perlin noise to the depth map. It is a 
% simplified implementation of the paper: 
% an image sunthesizer
% Ken Perlin, SIGGRAPH, Jul. 1985
% The bicubic interpolation is used, compared to the original version.
%
% Reference:
% HAZERD: an outdoor scene dataset and benchmark for single image dehazing
% IEEE International Conference on Image Processing, Sep 2017
% The paper and additional information on the project are available at:
% https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/
% If you use this code, please cite our paper.
%
% Input:
%   im: depth map
%   varargin{1}: decay term
% Output:
%   im: result of transmission with perlin noise added
%
% Authors: 
%   Yanfu Zhang: yzh185@ur.rochester.edu
%   Li Ding: l.ding@rochester.edu
%   Gaurav Sharma: gaurav.sharma@rochester.edu
% 
% Last update: May 2017


    [h,w] = size(im);
    i = 1;
    if nargin == 1
        decay = 2;
    else
        decay = varargin{1};
    end
    l_bound = min([h,w]);
    while i <= l_bound
        d = imresize(randn(i, i)*decay, size(im), 'bicubic');
        im = im+d;
        i = i*2;
    end
    im = mat2gray(im);
end
