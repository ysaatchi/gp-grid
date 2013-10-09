function [outaB] = pairVecMat(a,B)

if(isempty(B))
    outaB = a(:);
else
    a = a(:)';
    aext = repmat(a,size(B,1),1);
    Bext = repmat(B,length(a),1);
    outaB = [aext(:),Bext];
end
