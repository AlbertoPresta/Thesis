function scattered_image = scatter_dataset(set_path, name_new_folder,image_size, J,L,M)

ds = imageDatastore(set_path, "IncludeSubfolders",true,"LabelSource", "foldernames");
    
filt_opt.J =J;
filt_opt.L = L;
scat_opt.M = M;
scat_opt.oversampling = 0;
    
Wop = wavelet_factory_2d(image_size, filt_opt, scat_opt);
    
mkdir(strcat(name_new_folder));
existing_dirs = {};
    
c = 1;
d = 1;
    
while hasdata(ds)
    [I, info] = read(ds);
    label = char(info.Label);
    I = convertRGB_YUV(I);
    I = im2double(I);
    II = imresize(I,[100 100]);
    
    I1 = II(:,:,1);
    I2 = II(:,:,2);
    I3 = II(:,:,3);
    
    I1 = I1 - min(I1(:)) ;
    I1 = I1 / max(I1(:)) ;
    
    I2 = I2 - min(I2(:)) ;
    I2 = I2 / max(I2(:)) ;
    
    I3 = I3 - min(I3(:)) ;
    I3 = I3 / max(I3(:)) ;
    
    if ~any(strcmp(existing_dirs,label))
        existing_dirs{d} = label; 
        mkdir(strcat(name_new_folder,'/',label));
        
        d = d+1;
    end
    
    [scattered_image_1,~] = format_scat(scat(I1,Wop));
    [scattered_image_2,~] = format_scat(scat(I2,Wop));
    [scattered_image_3,~] = format_scat(scat(I3,Wop));
    
    scattered_image = [scattered_image_1 ; scattered_image_2 ; scattered_image_3];
    

    

    
    filename = strcat(int2str(c),'.mat');
    c = c+1;
    save(strcat(name_new_folder,'/',label,'/',filename),'scattered_image');
    
end
