train_path = fullfile('../../../data/train');
dir(train_path);
imdb = imageDatastore(train_path, "IncludeSubfolders",true,"LabelSource", "foldernames");




addpath '../../../../scatnet'
addpath_scatnet 

%%
tic
M = 2;
img_size = [100 ; 100];

for jj = 1:8
    disp(jj)
    J = 2;
    L = jj;
    fold = strcat('train/','prova_',string(J),'_',string(L),'_',string(M),'_');
    scattered_image = scatter_dataset(train_path, fold,img_size , J, L, M);
end
    
%% test data



test_path = fullfile('../../../data/valid');
dir(train_path);
imdb = imageDatastore(train_path, "IncludeSubfolders",true,"LabelSource", "foldernames");

tic
M = 2;
img_size = [100 ; 100];

for jj = 1:8
    disp(jj)
    J = 2;
    L = jj;
    fold = strcat('test/','prova_',string(J),'_',string(L),'_',string(M),'_');
    scattered_image = scatter_dataset(test_path, fold,img_size , J, L, M);
end

%%

size(scattered_image)
%min(scattered_image)

%%
disp('start')
ds = imageDatastore(train_path, "IncludeSubfolders",true,"LabelSource", "foldernames");


[I, info] = read(ds);
label = char(info.Label);
  
I = im2double(I);
II = imresize(I,[100 100]);
    
I1 = II(:,:,1);
I2 = II(:,:,2);
I3 = II(:,:,3);
    
u = zeros([100 100 3]);
u(:,:,1) = I1;
u(:,:,2) = I2;
u(:,:,3) = I3;

%% 
ds = fileDatastore('scattered_images','ReadFcn',@load, "IncludeSubfolders",true,"LabelSource", "foldernames",'FileExtensions','.mat');

%% open matlab file 
% compute scattering with 5 scales, 6 orientations
% and an oversampling factor of 2^2
x = uiuc_sample;
filt_opt.J = 5;
filt_opt.L = 6;
scat_opt.oversampling = 2;
Wop = wavelet_factory_2d(size(x), filt_opt, scat_opt);
Sx = scat(x, Wop);
% display scattering coefficients
image_scat(Sx)




