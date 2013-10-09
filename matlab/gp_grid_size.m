function [Cards] = gp_grid_size(xgrid,flip)

if(nargin < 2)
    flip = '';
end

D = length(xgrid);
Cards = zeros(1,D);
for d = 1:D
    Cards(d) = length(xgrid{d});
end

if(strcmp(flip,'flip'))
    Cards = fliplr(Cards);
end