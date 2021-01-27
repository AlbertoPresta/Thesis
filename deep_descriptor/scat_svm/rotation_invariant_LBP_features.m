function [pixelCounts, GLs] = rotation_invariant_LBP_features(I)

[rows, columns , ~] = size(I);

I1 = I(:,:,1);
I2 = I(:,:,2);
I3 = I(:,:,3);
%preallocate the image 
lbp_1 = [];
lbp_2 = [];
lbp_3 = [];

offset = 3;

for row = 2 : offset:rows - 1   
	for col = 2 : offset: columns - 1  
        
        % first channel lbp
		centerPixel = I1(row, col);
		pixel7=I1(row-1, col-1) > centerPixel;  
		pixel6=I1(row-1, col) > centerPixel;   
		pixel5=I1(row-1, col+1) > centerPixel;  
		pixel4=I1(row, col+1) > centerPixel;     
		pixel3=I1(row+1, col+1) > centerPixel;    
		pixel2=I1(row+1, col) > centerPixel;      
		pixel1=I1(row+1, col-1) > centerPixel;     
		pixel0=I1(row, col-1) > centerPixel; 
        
        pixels = [int8(pixel7) int8(pixel6) int8(pixel5) int8(pixel4) int8(pixel3) int8(pixel2) int8(pixel1) int8(pixel0)];
        eightBitNumber = find_maximum(pixels);
		%eightBitNumber = uint8(...
		%	pixel7 * 2^7 + pixel6 * 2^6 + ...
		%	pixel5 * 2^5 + pixel4 * 2^4 + ...
		%	pixel3 * 2^3 + pixel2 * 2^2 + ...
		%	pixel1 * 2 + pixel0);
		lbp_1 = [lbp_1, eightBitNumber ];
        
        % second channel lbp
		centerPixel = I2(row, col);
		pixel7=I2(row-1, col-1) > centerPixel;  
		pixel6=I2(row-1, col) > centerPixel;   
		pixel5=I2(row-1, col+1) > centerPixel;  
		pixel4=I2(row, col+1) > centerPixel;     
		pixel3=I2(row+1, col+1) > centerPixel;    
		pixel2=I2(row+1, col) > centerPixel;      
		pixel1=I2(row+1, col-1) > centerPixel;     
		pixel0=I2(row, col-1) > centerPixel;
        pixels = [int8(pixel7) int8(pixel6) int8(pixel5) int8(pixel4) int8(pixel3) int8(pixel2) int8(pixel1) int8(pixel0)];
        eightBitNumber = find_maximum(pixels);
		%eightBitNumber = uint8(...
		%	pixel7 * 2^7 + pixel6 * 2^6 + ...
		%	pixel5 * 2^5 + pixel4 * 2^4 + ...
		%	pixel3 * 2^3 + pixel2 * 2^2 + ...
		%	pixel1 * 2 + pixel0);
		lbp_2 = [lbp_2, eightBitNumber ];
        
        
        % third channel lbp
		centerPixel = I3(row, col);
		pixel7=I3(row-1, col-1) > centerPixel;  
		pixel6=I3(row-1, col) > centerPixel;   
		pixel5=I3(row-1, col+1) > centerPixel;  
		pixel4=I3(row, col+1) > centerPixel;     
		pixel3=I3(row+1, col+1) > centerPixel;    
		pixel2=I3(row+1, col) > centerPixel;      
		pixel1=I3(row+1, col-1) > centerPixel;     
		pixel0=I3(row, col-1) > centerPixel;
        pixels = [int8(pixel7) int8(pixel6) int8(pixel5) int8(pixel4) int8(pixel3) int8(pixel2) int8(pixel1) int8(pixel0)];
        eightBitNumber = find_maximum(pixels);
		%eightBitNumber = uint8(...
		%	pixel7 * 2^7 + pixel6 * 2^6 + ...
		%	pixel5 * 2^5 + pixel4 * 2^4 + ...
		%	pixel3 * 2^3 + pixel2 * 2^2 + ...
		%	pixel1 * 2 + pixel0);
		lbp_3 = [lbp_3, eightBitNumber ];
        
        
	end  
end 


l = [lbp_1, lbp_2, lbp_3];
[pixelCounts, GLs] = imhist(uint8(l));
pixelCounts = pixelCounts/norm(pixelCounts);


end