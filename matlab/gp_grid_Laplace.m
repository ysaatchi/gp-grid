function [nlml, dnlml] = gp_grid_Laplace(hypvec, lambda)
% calculate Laplace distribution
% hypvec contains log(theta) hyperparameters
%
%
% SANITY CHECK 
% gp_grid_Laplace - PASSED
% checkgrad('gp_grid_Laplace', hypers_init, 1e-3,lambda)
%
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(lambda==0)
    nlml=0;
    dnlml=zeros(size(hypvec));
    return
end
Q = length(hypvec);
theta = exp(hypvec);
nlml = lambda*sum(theta) - Q*log(lambda/2); % negative log likelihood

% Derivatives
dnlml =  lambda*theta; %dnlml/dlogtheta = dnlml/dtheta*dtheta/dlogtheta = lambda * theta 
