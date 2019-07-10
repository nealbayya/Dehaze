function [contDepth] = patchOcclusions(occludedDepth, k)
%slide a kxk kernel across occludedDepth and fills zero values with average
%of kernel

if mod(k,2) == 0 % keep kernel size odd
    k = k + 1;
end

contDepth = occludedDepth;
[rows, cols] = size(contDepth);
for i = 1: rows
    for j = 1:cols
        if contDepth(i,j) > 0
            continue
        end
        
        xl = i-floor(k/2);
        xr = i+floor(k/2);
        if i-floor(k/2) < 1
            xl = 1;
        elseif i+floor(k/2) > rows
            xr = rows;
        end
        yu = j-floor(k/2);
        yd = j+floor(k/2);
        if j-floor(k/2) < 1
            yu = 1;
        elseif j+floor(k/2) > rows
            yd = cols;
        end
        
        patch = contDepth(xl:xr, yu:yd);
        patch_values = patch(:);
        nonzero_values = patch_values(patch_values>0);
        patchAvg = mean(nonzero_values);
        
        contDepth(i,j) = patchAvg;
        
    end
end
        
            



end

