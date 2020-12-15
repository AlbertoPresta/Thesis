function featureVector = gaborFeatures(img)

% GABORFEATURES extracts the Gabor features of an input image.
% It creates a column vector, consisting of the Gabor features of the input
% image. The feature vectors are normalized to zero mean and unit variance.
%
%
% Inputs:
%       img         :	Matrix of the input image 
%       gaborArray	:	Gabor filters bank created by the function gaborFilterBank
%       d1          :	The factor of downsampling along rows.
%       d2          :	The factor of downsampling along columns.
%               
% Output:
%       featureVector	:   A column vector with length (m*n*u*v)/(d1*d2). 
%                           This vector is the Gabor feature vector of an 
%                           m by n image. u is the number of scales and
%                           v is the number of orientations in 'gaborArray'.
%
%
% Sample use:
% 
% img = imread('cameraman.tif');
% gaborArray = gaborFilterBank(5,8,39,39);  % Generates the Gabor filter bank
% featureVector = gaborFeatures(img,gaborArray,4,4);   % Extracts Gabor feature vector, 'featureVector', from the image, 'img'.
% 
% 
% 
%   Details can be found in:
%   
%   M. Haghighat, S. Zonouz, M. Abdel-Mottaleb, "CloudID: Trustworthy 
%   cloud-based and cross-enterprise biometric identification," 
%   Expert Systems with Applications, vol. 42, no. 21, pp. 7905-7916, 2015.
% 
% 
% 
% (C)	Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       PLEASE CITE THE ABOVE PAPER IF YOU USE THIS CODE.
disp('--------')
disp('immagine')
if (nargin ~= 1)        % Check correct number of arguments
    error('Please use the correct number of input arguments!')
end

%if size(img,3) == 3     % Check if the input image is grayscale
%    warning('The input RGB image is converted to grayscale!')
%    img = rgb2gray(img);
%end
gaborArray = gaborFilterBank(8,8,39,39); 

disp('-----')
c = gaborArray{1,1};
disp('????')
img = double(img);

img_r = img(:,:,1);
img_g = img(:,:,2);
img_b = img(:,:,3);


%% Filter the image using the Gabor filter bank

% Filter input image by each Gabor filter
[u,v] = size(gaborArray);
gaborResult_r = cell(u,v);
gaborResult_g = cell(u,v);
gaborResult_b = cell(u,v);
for i = 1:u
    for j = 1:v
        gaborResult_r{i,j} = imfilter(img_r, gaborArray{i,j});
        gaborResult_g{i,j} = imfilter(img_g, gaborArray{i,j});
        gaborResult_b{i,j} = imfilter(img_b, gaborArray{i,j});
    end
end


%% Create feature vector

% Extract feature vector from input image
featureVector_1_r = [];
featureVector_2_r = [];

featureVector_1_g = [];
featureVector_2_g = [];

featureVector_1_b = [];
featureVector_2_b = [];
for i = 1:u
    for j = 1:v
        
        gaborAbs_r = abs(gaborResult_r{i,j});
        gaborAbs_g = abs(gaborResult_g{i,j});
        gaborAbs_b = abs(gaborResult_b{i,j});

        gaborAbs_r = gaborAbs_r(:);
        gaborAbs_g = gaborAbs_g(:);
        gaborAbs_b = gaborAbs_b(:);

        % calcolo la media 
        mn_r = mean(gaborAbs_r);
        stdd_r =  std(gaborAbs_r,1); 
        
        mn_g = mean(gaborAbs_g);
        stdd_g =  std(gaborAbs_g,1);
        
        mn_b = mean(gaborAbs_b);
        stdd_b =  std(gaborAbs_b,1);
        
        featureVector_1_r =  [featureVector_1_r; mn_r];
        featureVector_2_r =  [featureVector_2_r; stdd_r];

        featureVector_1_g =  [featureVector_1_g; mn_g];
        featureVector_2_g =  [featureVector_2_g; stdd_g];
        
        featureVector_1_b =  [featureVector_1_b; mn_b];
        featureVector_2_b =  [featureVector_2_b; stdd_b];
        
        
        
        
    end
end


featureVector = [featureVector_1_r ; featureVector_2_r; featureVector_1_g ; featureVector_2_g; featureVector_1_b ; featureVector_2_b];



end


