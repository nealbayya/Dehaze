%Neal Bayya, October 2018
%Parse pfm file and generate depth map from disparity

% Are you going to use the training or test set?
imgset = 'training';
%imgset = 'test';

% Specify which resolution you are using for the stereo image set (F, H, or Q?)
imgsize = 'Q';
%imgsize = 'H';
%imgsize = 'F';

if strcmp(imgset,'training')
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
    ndisp = [290, 256, 640, 280, 280, 260, 260, 300, 330, 290, 290, 260, 240, 256, 760];
else
    image_names{1} = 'Australia';
    image_names{2} = 'AustraliaP';
    image_names{3} = 'Bicycle2';
    image_names{4} = 'Classroom2';
    image_names{5} = 'Classroom2E';
    image_names{6} = 'Computer';
    image_names{7} = 'Crusade';
    image_names{8} = 'CrusadeP';
    image_names{9} = 'Djembe';
    image_names{10} = 'DjembeL';
    image_names{11} = 'Hoops';
    image_names{12} = 'Livingroom';
    image_names{13} = 'Newkuba';
    image_names{14} = 'Plants';
    image_names{15} = 'Staircase';
    ndisp = [290, 290, 250, 610, 610, 256, 800, 800, 320, 320, 410, 320, 570, 320, 450];
end


for im_num = 1:15
    %read given files and disparity maps
    I0 = imread(['Middlebury/MiddEval3_input/', strcat(imgset,imgsize),'/',image_names{im_num},'/im0.png']);
    I1 = imread(['Middlebury/MiddEval3_input/', strcat(imgset,imgsize),'/',image_names{im_num},'/im1.png']);
    fileID = fopen(['Middlebury/MiddEval3_input/', strcat(imgset,imgsize),'/',image_names{im_num},'/calib.txt'],'r');
    calibtxt = splitlines(fscanf(fileID, '%c'));
    for i = 1: 12
        calib_assign = strsplit(calibtxt{i}, '=');
        fieldName = calib_assign{1};
        storename = 'value';
        eval([storename '= ' calib_assign{2}]);
        calib.(fieldName) = value;
    end
    calib.f0 = calib.cam0(1,1);
    calib.f1 = calib.cam1(1,1);
    disp0 = parsePfm(['Middlebury/MiddEval3_left/', strcat(imgset,imgsize),'/',image_names{im_num},'/disp0GT.pfm']);
    disp1 = parsePfm(['Middlebury/MiddEval3_right/', strcat(imgset,imgsize),'/',image_names{im_num},'/disp1GT.pfm']);
    fclose(fileID);
    
    %construct depth map
    
    depth0 = (calib.baseline * calib.f0 ) ./ (disp0 + calib.doffs);
    depth0 = patchOcclusions(depth0, 11);
    depth1 = (calib.baseline * calib.f1 ) ./ (disp1 + calib.doffs);
    depth1 = patchOcclusions(depth1, 11);
    %write files to locations in HazeRD
    
    im0name = ['IMG_', strcat(imgset,imgsize), '_', image_names{im_num}, '0_RGB.png'];
    imwrite(I0, fullfile('HazeRD', 'data', 'midd_img', im0name));
    im1name = ['IMG_', strcat(imgset,imgsize), '_', image_names{im_num}, '1_RGB.png'];
    imwrite(I1, fullfile('HazeRD', 'data', 'midd_img', im1name));
    
    depth0name = ['IMG_', strcat(imgset,imgsize), '_', image_names{im_num}, '0_depth.mat'];
    imDepth = depth0;
    save(fullfile('HazeRD', 'data', 'midd_depth', depth0name), 'imDepth');
    depth1name = ['IMG_', strcat(imgset,imgsize), '_', image_names{im_num}, '1_depth.mat'];
    imDepth = depth1;
    save(fullfile('HazeRD', 'data', 'midd_depth', depth1name), 'imDepth');
end


    
