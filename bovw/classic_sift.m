function [features, featureMetrics, varargout] = classic_sift(I)

I1 = single(I(:,:,1));
I2 = single(I(:,:,2));
I3 = single(I(:,:,3));
[~ ,features_1] = vl_sift(I1);
[~ ,features_2] = vl_sift(I2);
[~ ,features_3] = vl_sift(I3);
features_1 = double(features_1');
features_2 = double(features_2');
features_3 = double(features_3');
features = double([features_1 ; features_2 ; features_3]);

featureMetrics = var(features,[],2);
if nargout > 2
    % Return feature location information
    varargout{1} = multiscaleGridPoints.Location;
end
end