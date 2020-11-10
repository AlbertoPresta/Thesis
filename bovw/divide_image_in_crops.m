function [I,res] = divide_image_in_crops(pth)

I = imread(pth);
I = im2double(I);
disp(class(I))

sp = [1  200  400 600  800  ];
res = zeros(25,200,200,3);
cont = 1;

for i = 1:5
    cr_i = sp(i) + 200 -1 ;
    for j = 1:5
        cr_j = sp(j) + 200 -1;
        im = imcrop(I,[sp(i),sp(j),cr_i - sp(i)  ,cr_j-sp(j) ]);        
        filename = strcat('crops/image','_0',num2str(cont),'.jpg');
        
        res(cont,:,:,:) = im ;
        cont = cont +1 ;
        imwrite(im ,filename);

    
    end


end




end