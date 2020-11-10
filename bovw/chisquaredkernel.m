function kernel = chisquaredkernel(U,V)

[samples,~] = size(U);
kernel = zeros(samples,samples);
temp = [];
for r = 1:samples

    for c = r:samples
        tmp = chisquareddistance(U(r,:),U(c,:));
        kernel(r,c) = tmp;
        kernel(c,r) = tmp;
        temp = vertcat(temp,tmp);
    end
end
media = mean(temp);
kernel = exp(-(kernel)/media);

end


function res = chisquareddistance(u,v)

[k,~] = size(u);
indexes = [];
for i = 1:k
    if (u(i)~=0 || v(i)~=0) 
        indexes = vertcat(indexes,i);
    end
end

res = 0.5 * (sum( sqrt(u(indexes) - v(indexes)) / (u(indexes) + v(indexes))));





 
    
end