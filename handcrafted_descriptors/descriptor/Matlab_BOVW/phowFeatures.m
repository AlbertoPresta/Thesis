function [features, featureMetrics, varargout] = phowFeatures(I)

    
I = single(I);
%I= imresize(I,[100 100]);
%I1 = I(:,:,1);
%I2 = I(:,:,2);
%I3 = I(:,:,3);

%I1 = I1 - min(I1(:)) ;
%I1 = I1 / max(I1(:)) ;
    
%I2 = I2 - min(I2(:)) ;
%I2 = I2 / max(I2(:)) ;
    
%I3 = I3 - min(I3(:)) ;
%I3 = I3 / max(I3(:)) ;
    
%II = zeros([100 100 3]);
%II(:,:,1) = I1;
%II(:,:,2) = I2;
%II(:,:,3) = I3;
%II = single(II);
scales = [1.6 3.2 4.8 6.4 8.2]; % take into account different scale
step = 4;
col = 'Opponent';

[~, features] = vl_phow(I, 'Sizes',scales, 'Step', step, 'Color',col, 'FloatDescriptors',true);
features = features';

% use the variance of the SIFT as the feature metrics
featureMetrics = var(features, [], 2);


% Optionally return the feature location information. The feature location 
% information is used for image search applications. 

if nargout > 2
    varargout{1} = multiscaleGridPoints.Location;
end 
end


