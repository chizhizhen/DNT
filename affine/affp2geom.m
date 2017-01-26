function [ param_geom ] = affp2geom( p, sz )
% Thanks to Jongwoo Lim and David Ross for this code.  -- Shu Wang.

param_geom = [p(1), p(2), p(3)/sz(2), p(5), p(4)/p(3)*sz(2)/sz(1), 0];
% param = reshape(param, size(p));
if size(p,1)>1
    param_geom = param_geom(:);
end

end

