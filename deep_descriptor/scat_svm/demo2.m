train_path = fullfile('../../../patches/train');
dir(train_path);
tr = imageDatastore(train_path, "IncludeSubfolders",true,"LabelSource", "foldernames");
[imdsTrain,imdsTest] = splitEachLabel(tr,0.7,'randomized');
%%

save('training.mat','imdsTrain')
save('test.mat','imdsTest')
%%
addpath '../../../../scatnet'
addpath_scatnet 

%%
tic
M = 2;
img_size = [100 ; 100];

for jj = 1:8
    disp(jj)
    J = 4;
    L = jj;
    fold = strcat('scattered_image_lichen_',string(J),'_',string(L),'_',string(M),'_');
    scattered_image = scatter_dataset(train_path, fold,img_size , J, L, M);
    end


        
toc
disp('end')
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
    
filt_opt.J =4;
filt_opt.L = 4;
scat_opt.M = 4;
scat_opt.oversampling = 0;
    
Wop = wavelet_factory_2d([100;100], filt_opt, scat_opt);



[scattered_image_1,~] = format_scat(scat(I1,Wop));
disp(size(scattered_image_1))
disp('end')
%% 
ds = fileDatastore('scattered_images','ReadFcn',@load, "IncludeSubfolders",true,"LabelSource", "foldernames",'FileExtensions','.mat');

%% open matlab file 

% and divide to train svm (after computing)

file_list = ds.Files;
[m,n] = size(file_list);

data = zeros(810,75)
label = zeros(810,1)
for i = 1:m
    dt = file_list{i}.scattered_image;
    c = sum(sum(dt,2),3);
    if i == 1
        disp(size(c))
    end
    
    
end








