brickWall = imread('bricks.jpg');
rotatedBrickWall = imread('bricksRotated.jpg');
carpet = imread('carpet.jpg');

%rickWall = single(brickWall);
%brickWall = imresize(brickWall,[100 100]);
%%
I=imread('gatto.jpg');
I = I(:,:,1);
mapping=getmapping(8,'u2'); 
H1=descriptor_LBP(I,1,8,mapping,'nh');
%%

img = imread('gatto.jpg');
imgg = single(img);

imgg = imresize(imgg, [500,500]);
[pixelCounts, GLs] =LBP_features(imgg);

%%

imrot = imrotate(img, 90);
%immagineruot = imresize(imggrot, [500,500]);
imggrot = single(imrot);
imggrot = imresize(imggrot, [500,500]);
[r, GLs] = LBP_features(imggrot);

%%

figure 
subplot(1,2,1)
bar(pixelCounts)
%make here your first plot
subplot(1,2,2)
bar(r)



%%
im = imread('gatto.jpg');
im = imresize(im, [500,500]);
figure 
subplot(2,2,1)
image(im)
xlabel('original image')

%make here your first plot
subplot(2,2,2)
bar(pixelCounts)
xlabel('LBP histogram')
ylabel('frequency')
%make here your second plot

subplot(2,2,3)
im = imread('gatto.jpg');
im = imresize(im, [500,500]);
im = imrotate(im, 90);
image(im)
xlabel('rotated image')

%make here your first plot
subplot(2,2,4)
bar(r)
xlabel('LBP histogram of rotated image')
ylabel('frequency')
%make here your second plot




%%


bar(GLs, pixelCounts/norm(pixelCounts));
%%
img = imread('prova.jpg');
img = single(img);
img = imresize(img, [100 100]);
features1 = extractLBPFeatures(img(:,:,1),'Upright',false);
features2 = extractLBPFeatures(img(:,:,2),'Upright',false);
features3 = extractLBPFeatures(img(:,:,3),'Upright',false);



f = [features1, features2, features3];


[pixelCounts, GLs] = imhist(uint8(f));
bar(GLs, pixelCounts);

%%

c = [128 64 32 16 8 4 2 1];

c = circshift(c,1);
