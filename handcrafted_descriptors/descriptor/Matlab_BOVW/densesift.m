function [features, featureMetrics, varargout] = densesift(I)

    
I = single(I);
I= imresize(I,[100 100]);
I1 = I(:,:,1);
I2 = I(:,:,2);
I3 = I(:,:,3);

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
%scales = [1.6 3.2 4.8 6.4 ]; % take into account different scale
step = 8;
magnif = 3;
%Is = vl_imsmooth(I, sqrt((step/magnif)^2 - .25)) ;
I1 = single(I1);
I2 = single(I2);
I3 = single(I3);
[~, features_1]  = vl_dsift(I1, 'size', step) ;
[~, features_2]  = vl_dsift(I2, 'size', step) ;


[~, features_3]  = vl_dsift(I3, 'size', step) ;
features_1 = features_1';
features_2 = features_2';
features_3 = features_3';

features = [features_1 features_2 features_3];

features = single(features);


% use the variance of the SIFT as the feature metrics
featureMetrics = var(features, [], 2);


% Optionally return the feature location information. The feature location 
% information is used for image search applications. 

if nargout > 2
    varargout{1} = multiscaleGridPoints.Location;
end 
end