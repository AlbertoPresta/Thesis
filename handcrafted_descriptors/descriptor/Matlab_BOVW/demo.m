clear all 
clc
%% da scaricare vlfeat
run('../vlfeat/toolbox/vl_setup')
%%
if isempty(gcp)
     parpool;
end

%% 
setdir_train  = fullfile('..','..','..','..','data','train');
setdir_test  = fullfile('..','..','..','..','data','valid');

imds_train = imageDatastore(setdir_train, 'IncludeSubFolders', true, 'LabelSource','foldernames');
imds_valid = imageDatastore(setdir_test, 'IncludeSubFolders', true, 'LabelSource','foldernames');

   
labels = cellstr(unique(imds_train.Labels));
    train_lab = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    lab_dict = containers.Map(labels, train_lab);


 


%%
tic
vocsize = [500];

for vc = vocsize

    disp('------------------------------')
    disp(vc)
    extractor = @scalepropagationsift; 
    bag = bagOfFeatures(imds_train, 'CustomExtractor',extractor, 'VocabularySize',vc, 'StrongestFeatures',1);
    
    savename = fullfile(strcat('bag_of_words_scale_image8'));
    save(savename, 'bag'); 
end



% create python descriptors 
vis = [500];
%validdir = fullfile('..','..','..','..','mnt','workdata','presta','data','valid');
%validimds = imageDatastore(validdir, 'IncludeSubFolders', true, 'LabelSource','foldernames');
for v = vis
   
    clc
    disp('*********************************************')
    disp(v)
    disp('***********************************************')
    vis = v;
    visstr = string(vis);
    bag =  load(fullfile(strcat('bag_of_words_scale_image8')));
    bag = bag.bag

    labels = cellstr(unique(imds_train.Labels));
    train_lab = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    lab_dict = containers.Map(labels, train_lab);



    folder = fullfile(strcat('image_',visstr,'_8'));

           
              
    [train_features, training_labels] = extract_feature_for_python(imds_train, bag, vis, folder, 0,lab_dict);

    disp('------------------------------------------------------------------------------------')
    [tst, tst_lbs] = extract_feature_for_python(imds_valid, bag, vis, folder, 1,lab_dict);
    

                                
end
                           


