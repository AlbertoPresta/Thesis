
clear all;
clc;


setdir_train = '/Users/admin/Desktop/tesi/data/train';
imds_train = imageDatastore(setdir_train, 'IncludeSubFolders',true,'LabelSource','foldernames');

setdir_test = '/Users/admin/Desktop/tesi/data/valid';
imds_test = imageDatastore(setdir_test, 'IncludeSubFolders',true,'LabelSource','foldernames');

path = '/Users/admin/Desktop/tesi/Thesis/handcrafted_descriptors/gabor_classification/descriptors_calculation/gabor';
addpath(path);


%[trainingSet, validationSet] = splitEachLabel(imds_train, 0.70, 'randomize' );

% create dictionary of labels 
labels= cellstr(unique(imds_train.Labels));
train_lab = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
lab_dict = containers.Map(labels,train_lab);

%% CREATION OF THE DESCRIPORS

train_pth = imds_train.Files;
lab_pth = cellstr(imds_train.Labels);

img_filename = '../dsc/training_descriptors';
lab_filename = '../dsc/training_labels' ;


%[dsc,lab] = calculate_descriptors(train_pth, lab_pth,img_filename, lab_filename,lab_dict);



%% CREATION OF THE DESCRIPORS

test_pth = imds_test.Files;


lab_pth = cellstr(imds_test.Labels);

img_filename = 'dsc/test_descriptors';
lab_filename = 'dsc/test_labels' ;


[dsc,lab] = calculate_descriptors(test_pth, lab_pth,img_filename, lab_filename,lab_dict);


%%

orientation = [1 2 3 4 5 6 7 8];
scales = [1 2 3 4 5 6 7 8]; 
test_pth = imds_test.Files;


train_pth = imds_train.Files;
lab_pth_train = cellstr(imds_train.Labels);
lab_pth_test = cellstr(imds_test.Labels);
for i=7:8
    for j = 1:8
        disp('-----')
        disp(i)
        disp(j)
        disp('-------')
        or = orientation(i);
        sc = scales(j);
        img_filename = strcat('dsc/',num2str(i),'_',num2str(j),'_','train_descriptors');
        lab_filename = strcat('dsc/',num2str(i),'_',num2str(j),'_','train_labels');
        
        [dsc,lab] = calculate_descriptors(train_pth, lab_pth_train,img_filename, lab_filename,lab_dict,or,sc);
        
        img_filename = strcat('dsc/',num2str(i),'_',num2str(j),'_','test_descriptors');
        lab_filename = strcat('dsc/',num2str(i),'_',num2str(j),'_','test_labels');
        
        [dsc,lab] = calculate_descriptors(test_pth, lab_pth_test,img_filename, lab_filename,lab_dict,or,sc);


    end
end


