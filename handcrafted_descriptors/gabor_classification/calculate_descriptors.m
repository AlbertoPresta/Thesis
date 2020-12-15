function [dsc,lab] = calculate_descriptors(train_pth, lab_pth,img_filename, lab_filename, lab_dict)

    m =size(train_pth,1);
    
    dsc = zeros(m,384);
    lab = zeros(m,1);
    for i=1:m
        pth = train_pth{i};
        I = imread(pth);
        featureVector = gaborFeatures(I);
        dsc(i,:) = featureVector;
        lab(i) = lab_dict(lab_pth{i});
        disp('---')
        disp(lab_dict(lab_pth{i}))
    end

    save(img_filename, 'dsc', '-v7');
    save(lab_filename,'lab','-v7');



end