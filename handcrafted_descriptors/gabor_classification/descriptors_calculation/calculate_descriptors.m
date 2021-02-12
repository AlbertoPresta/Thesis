function [dsc,lab] = calculate_descriptors(train_pth, lab_pth,img_filename, lab_filename, lab_dict,orient, scales)

    m =size(train_pth,1);
    
    dsc = zeros(m,orient*scales*6);
    lab = zeros(m,1);
    for i=1:m
        
        pth = train_pth{i};
        %disp(pth)
        I = imread(pth);
        featureVector = gaborFeatures(I,orient, scales);
        dsc(i,:) = featureVector;
        lab(i) = lab_dict(lab_pth{i});

    end

    save(img_filename, 'dsc', '-v7');
    save(lab_filename,'lab','-v7');



end