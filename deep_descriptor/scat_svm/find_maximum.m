function bitnum = find_maximum(pixels)

c = [2^7 2^6 2^5 2^4 2^3 2^2 2^1 2^0];

c = single(c);
pixels = single(pixels);
res = zeros(8,1);

for i = 1:8
    c = circshift(c,1);
    res(i) = dot(pixels,c);
end

bitnum = uint8(min(res));


end