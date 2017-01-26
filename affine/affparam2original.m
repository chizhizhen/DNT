function q = affparam2original(p,block_size)
%% Copyright (C) Shu Wang.
%% All rights reserved.
% this function transform a param.est(p) to the original parameter q [cx cy scalex scaley theta]

temp_param = affparam2geom(p);
q = [temp_param(1),temp_param(2),temp_param(3)*block_size(2),temp_param(5)*temp_param(3)*block_size(1),temp_param(4)];