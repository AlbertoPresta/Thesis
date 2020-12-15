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

disp('hi')
N = row * col ;
[F1, ~] = vl_sift(gray_img);
%[F2, ~] = vl_sift(sImage2);
%[F3, ~] = vl_sift(sImage3);
proptype = 'geometric';
weight_func = 'linear';

if strcmp(proptype, 'geometric')
    newF1 = propagateScales(proptype, F1, size(gray_img), weight_func);

else if strcmp(proptype, 'image-aware')
        newF1 = propagateScales(proptype, F1, gray_img, weight_func);

    end
end
features_1 = dense_sift(sImage1, newF1);
features_2 = dense_sift(sImage2, newF1);
features_3 = dense_sift(sImage3, newF1);


%features_1 = reshape(features_1, [N 128]);
%features_2 = reshape(features_2, [N 128]);
%features_3 = reshape(features_3, [N 128]);

features = cat(2,features_1,features_2,features_3);
features = double(features);

featureMetrics = var(features,[],2);





if nargout > 2
    % Return feature location information
    varargout{1} = multiscaleGridPoints.Location;
end
end



function desc = dense_sift(image, frames)

pad_size = 2 * 16;
image_padded = padarray(image, [pad_size pad_size], 'replicate');
frames(1, :) = frames(1, :) + pad_size;
frames(2, :) = frames(2, :) + pad_size;

[f1, desc] = vl_sift(image_padded, 'frames', frames, 'magnif', 1);
[~, ind] = sortrows(f1');
desc = desc(:, ind)';
disp('print size of desc')
indexes = floor(linspace(1,40000,40000/8)');
disp(size(desc))
desc = desc(indexes,:);
disp(size(desc))
%desc = reshape(desc(:), [size(image, 1) size(image, 2) 128]);

end