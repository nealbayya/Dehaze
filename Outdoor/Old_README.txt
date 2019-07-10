% This README file accompanies the HazeRD dataset described in the paper:
% Y. Zhang, L. Ding, G. Sharma, "HazeRD: an outdoor scene dataset and benchmark for Single image dehazing", 
% IEEE International Conference on Image Processing, Sep. 2017
% The paper and additional information on the project are available at:
% https://labsites.rochester.edu/gsharma/research/computer-vision/hazerd/
% The HazeRD distribution includes outdoor color+depth image pairs and MATLAB code for generating simulated hazy images
% for different haze conditions and code for evaluating dehazed images with respect to the ground truth.
% See instructions below for installing and running the code.
%
% The code is copyrighted by the authors. Permission to copy and use
% this software and datasets for noncommercial use is hereby granted provided this
% notice is retained in all copies and the paper and the distribution
% and HazeRD publications are clearly cited.
%
% The software code is provided "as is" with ABSOLUTELY NO WARRANTY expressed or
% implied. Use at your own risk.
%
% Contacts:
% Yanfu Zhang: yzh185@ur.rochester.edu
% Li Ding: l.ding@rochester.edu
% Gaurav Sharma: gaurav.sharma@rochester.edu
%
% DEPENDENCIES/EXTERNAL CONTRIBUTORS:
% The RGB-D images used for generating HazeRD dataset are obtained based on the approach described in:
% L. Ding and G. Sharma, “Fusing Structure from Motion and Lidar for Dense Accurate Depth Map Estimation,”
% IEEE International Conference Acoustic, Speech, and Signal Processing, Mar. 2017.
% The code for computing CIEDE2000 color difference metric is based on the paper: 
% G. Sharma, W. Wu, E. N. Dalal, "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations," 
% Color Research and Application, vol. 30, no. 1, pp. 21-30, Feb. 2005%

I) Installation:

Unzip the source code archive. This will create a sub-directory "HazeRD"
which is intended to be the directory where you run the code. 

II) Executing the code:

Start MATLAB and change to the just created "HazeRD" subdirectory. 


II(a) HazeRD Dataset Generation:

To generate the HazeRD dataset run demo_simu_haze.m.
The synthesized hazy images used for the benchmarking in the above mentioned HazeRD paper will be stored in the subdirectory ./data/simu/. 
The hazy images are named as {scene name}_{visual range} as png files, and the transmission maps are named as {scene name}_{visual range} as mat files,. 
The haze-free RGB images are stored in ./data/img/ and the depth maps in ./data/depth/, with names that clearly establish the correspondence.

II(b) Computing fidelity metrics for dehazed images with respect to originals:

IMPORTANT: 
The scipts expect the input for the dehazed image and the ground truth using the same visual range,  
otherwise the code may not work properly.

The script demo_metrics.m demonstrates the computation of fidelity metrics for dehazed images. The script includes the 
names (along with paths, and following the above convention) of a sample dehazed image, the corresponding estimated transmission, the ground truth transmission and the original ground truth RGB-D 
image based on which the metrics are computed. These would need to be updated to compute metrics for results obtained with other methods (code for dehazing is not included here). 
The script provides as output the SSIM and CIEDE2000 error metrics between the ground truth RGB image and 
the dehazed  RGB image. Additionally, the MSE between the ground truth and estimated transmission map is also reported.




File/Directory structure
HazeRD
|-- data
|   |-- dehaze
|   |   |-- contains dehazed image and estimated parameters
|   |-- depth
|   |   |-- contains 10 depth maps (mat files)
|   |-- img
|   |   |-- contains 10 color images (jpg files)
|   |-- simu
|   |   |-- folder for simulated hazy images
|-- utils
|   |-- deltaE2000.m
|   |-- hazy_simu.m
|   |-- perlin_noise.m
|-- demo_simu_haze.m
|-- demo_metrics.m
|-- README.txt
