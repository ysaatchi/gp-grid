% makePossibleComb
% 
% returns all the element wise combinations of a set of vectors. 
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [outAB] = makePossibleComb(Vecs)

D = length(Vecs);

outAB = [];
for d = D:-1:1
   
    outAB = pairVecMat(Vecs{d},outAB);
    
end

