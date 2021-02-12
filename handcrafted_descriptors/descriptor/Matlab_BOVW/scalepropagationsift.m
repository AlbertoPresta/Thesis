function [features,featureMetrics,varargout] = scalepropagationsift(I)
% proptype can be
%
%   'geometric'     - propagate scale information from detected interest
%       points by considering only the spatial locations where scales were
%       detected.
%   'image-aware'   - scales are propagated as above, but using image
%       intensities in order to guide scale propagation.
%   'match-aware'   - consider the two images, propagating only the scales
%       of pixels that were selected as (sparse) key-points in both images.
%   see [1] for more details.
%
% weight_func can be
%   'exponential'       - Used exponential weights.
%   'linear' (default)  - Used the linear weights.
%
if size(I, 3) == 3
    gray_img = rgb2gray(I);
end

[row,col,~] = size(I);

gray_img = im2single(gray_img);
sImage1 = im2single(I(:,:,1));
sImage2 = im2single(I(:,:,2));
sImage3 = im2single(I(:,:,3));


N = row * col ;
[F1, ~] = vl_sift(sImage1);
[F2, ~] = vl_sift(sImage2);
[F3, ~] = vl_sift(sImage3);
proptype = 'image-aware';
weight_func = 'linear';

if strcmp(proptype, 'geometric')
   newF1 = propagateScales(proptype, F1, size(gray_img), weight_func );
   newF2 = propagateScales(proptype, F2, size(gray_img), weight_func );
   newF3 = propagateScales(proptype, F3, size(gray_img), weight_func );

else if strcmp(proptype, 'image-aware')
   newF1 = propagateScales(proptype, F1, sImage1, weight_func);
   newF2 = propagateScales(proptype, F2, sImage2, weight_func);
   newF3 = propagateScales(proptype, F3, sImage3, weight_func);

    end
end
features_1 = dense_sift(sImage1, newF1);
features_2 = dense_sift(sImage2, newF2);
features_3 = dense_sift(sImage3, newF3);




features = cat(2,features_1,features_2,features_3);
features = double(features);

featureMetrics = var(features,[],2);





if nargout > 2
    % Return feature location information
    varargout{1} = multiscaleGridPoints.Location;
end
end



function desc = dense_sift(image, frames)

pad_size = 2 * 4;
image_padded = padarray(image, [pad_size pad_size], 'replicate');
frames(1, :) = frames(1, :) + pad_size;
frames(2, :) = frames(2, :) + pad_size;

[~, desc] = vl_sift(image_padded, 'frames', frames, 'magnif', 1);



desc = desc';
indexes = floor(linspace(1,40000,40000/8)');
desc = desc(indexes,:);


end