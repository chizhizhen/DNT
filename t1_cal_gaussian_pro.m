function probability = t1_cal_gaussian_pro(data, cent, sig)
%% Copyright (C) Shu Wang.
%% All rights reserved.
% motion model based on gaussian distribution
datasize = size(data);
probability = zeros(1,datasize(2));
if length(cent)==1
    for i = 1:datasize(2)
        probability(i) = (2*pi*sig)^(-.5) * exp(-0.5*(data(i) - cent)^2/sig);
    end
else
    for i = 1:datasize(2)
        probability(i) = (2*pi*sig)^(-.5) * exp(-0.5*(data(:,i) - cent)'*(data(:,i) - cent)/sig);
    end
end