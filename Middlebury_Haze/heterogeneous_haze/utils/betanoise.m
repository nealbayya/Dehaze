function [beta, lframe, lf_superpos] = betanoise(beta)
%betanoise: Induces noise in beta
%defines random gaussian beta amplifications or attenuations on larger
%frame.  superposition of gaussian models induces noise in beta map.

%Input: beta as a 2-D double matrix
%Output: noisy beta map

[sx, sy] = size(beta);
%% Internal Parameters
lfx = 3*sx;
lfy = 3*sy;
maxPatches = 3;
filtersize = 2*sx;
gaussmult = 25*filtersize;
%% Prepare frame
filterhalf = floor(filtersize/2);
lframe = ones(lfx, lfy);
npatches = randi(maxPatches+1)-1;

for i=1:npatches
    x = randi(lfx-filtersize)+filterhalf;
    y = randi(lfy-filtersize)+filterhalf;
    
    xmin = x-filterhalf;
    xmax = x+filterhalf;
    ymin = y-filterhalf;
    ymax = y+filterhalf;
    
    densify = rand;
    gaussian = fspecial('gaussian',length(xmin:xmax), filtersize/10).*gaussmult;
        
    if densify > 0.5
        gaussian = gaussian + 1;
    else
        gaussian = 1-gaussian;
    end
        

    lframe(xmin:xmax, ymin:ymax) = lframe(xmin:xmax, ymin:ymax).*gaussian;
    
end

tlx = floor(lfx/2) - floor(sx/2);
tly = floor(lfy/2) - floor(sy/2);

lf_superpos = lframe(tlx: tlx+sx-1, tly: tly+sy-1);

beta = beta.*lf_superpos;

end

