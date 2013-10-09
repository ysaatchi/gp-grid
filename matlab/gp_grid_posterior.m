function [nlml, dnlml] = gp_grid_posterior(x, likefun, priorfuns)

% priorfuns - a structure .indices = hyperparameter indices, .func =
% @priorfunc()
%
%
% SANITY CHECK - gradient of cov function,
% gp_grid_posterior - PASSED
% checkgrad('gp_grid_posterior', hypers_init, 1e-3, loglikefunc, logpriorfuns)
%
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(nargin<3)
    P=0;
else
    P = length(priorfuns);
end
Q = length(x);
nlmlv = zeros(1,P+1);
dnlmlv = zeros(Q,P+1);
[nlmli, dnlmli] = likefun(x);
nlmlv(1)=nlmli;
dnlmlv(:,1) = dnlmli;
for pi = 1:P
    hyps_indx_for_prior = priorfuns{pi}.indices;
    [nlmli, dnlmli] = priorfuns{pi}.func(x(hyps_indx_for_prior));
    nlmlv(pi+1)=nlmli;
    dnlmlv(hyps_indx_for_prior,pi+1) = dnlmli;
end
nlml = real(sum(nlmlv));
dnlml = real(sum(dnlmlv,2));