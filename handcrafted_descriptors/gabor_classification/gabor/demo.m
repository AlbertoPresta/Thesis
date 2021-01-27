train_path = fullfile('../../../patches/train');
dir(train_path);
imdb = imageDatastore(train_path, "IncludeSubfolders",true,"LabelSource", "foldernames");
%imdb.images.data = (imdb.images.data-min(imdb.images.data(:)))/(max(imdb.images.data(:)-min(imdb.images.data(:))));
%imageMean = mean(imdb.images.data,3); 
%for j= 1:size(imdb.images.data,3)
%    imdb.images.data(:,:,j) = imdb.images.data(:,:,j) - imageMean ;
% end
[imdsTrain,imdsTest] = splitEachLabel(imdb,0.7);