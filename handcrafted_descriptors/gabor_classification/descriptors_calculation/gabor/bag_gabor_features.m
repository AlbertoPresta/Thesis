function  [features, featureMetrics, varagout] = bag_gabor_features(I)

features = gaborFeatures(I);

featureMetrics = var(features, [], 2);

if nargout > 2
    varagout{1} = multiscaleGridPoints.Location;
end 


end