function [ param ] = affparam2ultimate( p ,block_size)
%% Copyright (C) Shu Wang.
%% All rights reserved.

param0 = [p(1), p(2), p(3)/block_size(2), p(5), p(4)/p(3)*block_size(2)/block_size(1), 0]';
param = affparam2mat(param0);
