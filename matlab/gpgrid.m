%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% gpgrid()
%
% Fast version of GP prediction ALL points lie on a grid (with G grid points per dim).
% This version also attempts to do everything in linear memory complexity.
% gpgrid_img_dn does not consider spherical noise as a hyperparameter to be learned,
% but as input dependent noise specified by the user.
%
%
% Usage: [hypers_learned, nlml, Iout, Vout] = gpgrid(input, noise, xstar, InitParamSet, cov, hyps_in_dim)
%        [hypers_learned, nlml] = gpgrid(input, noise, xstar, InitParamSet, cov, hyps_in_dim)
%
% Inputs:
%     input         Input structure for GP-grid
%       .index_to_N	Indices of not dummy locations
%       .data       The sub image to be interpolated
%       .xgrid         Cell array of per dimension grid points
%     noise_struct
%       .learn      flag if to learn noise hyperparameter
%       .var        noise variance of observations. If is a vector and
%                   noise_struct.learn == 1, then will only learn the noise
%                   for not dummy locations. if noise_struct.learn == 0,
%                   then noise_struct.var contains the known noise
%                   variance.
%       .sphericalNoise
%                   learn a single noise hyperparameter for entire input
%                   space (spherical noise).
%     mask          Binary mask for output image
%     InitParamSet  Values of initial parameters.
%       .learn      If true, then use initial values to find optimal hypers.
%                   If false, then use the first row of .vals as the
%                   optimal parameters (no learning).
%       .vals       Matrix where each row
%                   is an initial guess for all hypers. If learning noise
%                   hyperparameter, then the noise hyperparameter must be
%                   located a the end of the hyperparameter vals vector
%     cov           Covariance function as in gpml-matlab
%     hyps_in_dim   A cell vector [size D]. Each cell contains a vector of
%                   the hyperparameters in the corresponding dimension. The
%                   parameters must be in the order to be used in the
%                   covariance function.
%
% Outputs:
%     hypers_learned  learned hyperparameters
%     Iout              interpolated Image
%     Vout              GP variance (not implemented)
%
%
% Note:
% To use the package you must first install gpml-matlab.
% The package can be found in www.gaussianprocess.org/gpml/. The last
% version tested was gpml-matlab-v3.1-2010-09-27
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [hypers_learned, results, Iout, Vout] = gpgrid(input, gpmodel, xstar, itrnum)

if(nargin < 4)
    itrnum = 1000;
end

% % allocate memory
% Iout = zeros(IoutSize(:)');
% If only one output than only return learned hyperparameters
% Otherwise, also perform prediction for Iout
if(nargout > 2)
    predictionFlag = true;
    if(nargout > 3)
        voutFlag = true;
    else
        voutFlag = false;
    end
else
    predictionFlag = false;
end
if(~isfield(gpmodel,'logpriorfuns')) % if no logfunc specified than use standard likelihood function
    gpmodel.logpriorfuns = []; %@(t) gp_grid_prodcov( t, input.xgrid, input, noise_struct, cov, hyps_in_dim);
end

% % y is a column stack of Isub
% y = input.data(:);
% % D is the number of dimensions (should be 2 for Images)
% D = length(input.xgrid);
results = [];

% prepare a results table with a row for every run, first column will be the
% nlml value and the rest will be the values of the optimal hyperparameters.
allResultTable = zeros(size(gpmodel.hyperparams,1),size(gpmodel.hyperparams,2)+2);

loglikefunc = @(t) gp_grid_prodcov( t, input, gpmodel);
logpostfunc = @(t) gp_grid_posterior(t,loglikefunc,gpmodel.logpriorfuns);


if(isempty(gpmodel.learn) || gpmodel.learn==true)
    %     parfor run_i = 1:size(InitParamSet.vals,1)
    for run_i = 1:size(gpmodel.hyperparams,1)
        hypers_init = gpmodel.hyperparams(run_i,:)';
        
        p.length =    -itrnum;
        p.method =    'BFGS';% 'BFGS' 'LBFGS' or 'CG'
        p.verbosity = 1; %0 quiet, 1 line, 2 line + warnings (default), 3 graphical
%         p.mem        % number of directions used in LBFGS (default 100)
        tic
        [hypers_learned fx, numofitr] = minimize_new(hypers_init, logpostfunc, p );
        toc
        %         disp(hypers_learned);
        allResultTable(run_i,:) = [fx(end),numofitr,hypers_learned'];
        save('gp_grid_temp_hypers','allResultTable')
    end
    sortResultTable = sortrows(allResultTable,1);
    hypers_learned = sortResultTable(1,3:end)';
    results.nlml = sortResultTable(1,1);
    results.numofitr = sortResultTable(1,2);
else
    hypers_learned = gpmodel.hyperparams(1,:)';
    results.nlml = logpostfunc(hypers_learned);
end

% disp(['sf learned = 2^',num2str(imin-1),' ells learned = 2^',num2str(imin_learnedtt(imin)-1)]);
% disp((hypers_learned)./log(2))
% disp(exp(hypers_learned))

% pause
if(predictionFlag)
    
    [nlml, ~, alpha_kron, Qs, V_kron] = loglikefunc( hypers_learned );
    
    %learnexectime =toc
    disp(nlml);
    % if the noise model was spherical and the hyperparamter was learned, then
    % get the new noise matrix
    if(gpmodel.noise_struct.learn == true)
        gamma2 = exp(2*hypers_learned(end));
        gpmodel.noise_struct.var = gpmodel.noise_struct.var*gamma2;
    end
    
    
    % use meshgid to create a locations for interpolation
    % we interpolate also observed locations for denoising
    
    % allocate memory for variance of prediction
    %     Iout = zeros(size(IoutSize));
    %     Vout = zeros(size(IoutSize));
    
    % perform prediction using covMatern covariance function
    if(voutFlag)
        [mu_f, var_f] = gpr_cov_grid_predict_parallel(xstar, hypers_learned, input, gpmodel, alpha_kron, V_kron, Qs);
    else
        [mu_f] = gpr_cov_grid_predict_parallel(xstar, hypers_learned, input, gpmodel, alpha_kron, V_kron, Qs);
        var_f = 0;
    end
    Iout = mu_f;
    Vout = var_f;
    
end



end