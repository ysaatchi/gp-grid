% sub2ind_highD
%
% calculate the index for any highdimensional subscripts. 
%
% Cards - vector of cardinality of each dimension
% Subs - matrix of subscripts. Each row corresponds to a single set of
%           subscripts
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [index] = sub2ind_highD(Cards, Subs)

    if(length(Cards) ~= size(Subs,2))
        error('dimensions must agree');
    end
    if(sum(sum(Subs>repmat(Cards(:)',size(Subs,1),1))+sum(Subs<1))>0)
        error('subs out of bound');
    end
    D = length(Cards);
    M = ones(D,1);    
    
    for d = 2:D
       M(d) = Cards(d-1)*M(d-1);
    end
    
    subs_m1 = Subs-1;
    index = subs_m1*M+1;

end