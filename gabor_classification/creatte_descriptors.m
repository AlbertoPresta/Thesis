
clear all;
clc;


setdir = '/Users/admin/Documents/MATLAB/patches/train';
imds = imageDatastore(setdir, 'IncludeSubFolders',true,'LabelSource','foldernames');
disp('end')


path = '/Users/admin/Desktop/tesi/Thesis/gabor_classification/gabor';
addpath(path);


[trainingSet, validationSet] = splitEachLabel(imds, 0.70, 'randomize');

% create dictionary of labels 
labels= cellstr(unique(imds.Labels));
train_lab = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
lab_dict = containers.Map(labels,train_lab);

%% CREATION OF THE DESCRIPORS

train_pth = trainingSet.Files;
lab_pth = cellstr(trainingSet.Labels);

img_filename = 'dsc/training_descriptors';
lab_filename = 'dsc/training_labels' ;


[dsc,lab] = calculate_descriptors(train_pth, lab_pth,img_filename, lab_filename,lab_dict);



%% CREATION OF THE DESCRIPORS

train_pth = validationSet.Files;
lab_pth = cellstr(validationSet.Labels);

img_filename = 'dsc/test_descriptors';
lab_filename = 'dsc/test_labels' ;


[dsc,lab] = calculate_descriptors(train_pth, lab_pth,img_filename, lab_filename,lab_dict);











