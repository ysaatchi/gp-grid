%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% gpgrid_dn()
%
% Fast version of GP prediction ALL points lie on a grid (with G grid points per dim).
% This version also attempts to do everything in linear memory complexity.
% gpgrid_img_dn does not consider spherical noise as a hyperparameter to be learned,
% but as input dependent noise specified by the user.
%
%
% Usage: [hypers_learned, Iout, Vout] = gpgrid_dn(input, Xgrid, noise, xstar, InitParamSet, cov, hyps_in_dim)
%        [hypers_learned] = gpgrid_dn(input, Xgrid, noise, xstar, InitParamSet, cov, hyps_in_dim)
%
% Inputs:
%     input         Input structure for GP-grid    
%       .index_to_N	Indices of not dummy locations
%       .data       The sub image to be interpolated
%     Xgrid         Cell array of per dimension grid points
%     noise_struct
%       .learn      flag if to learn noise hyperparameter
%       .var        noise variance of observations. If this is a vector and
%                   noise_struct.learn == 1, then will only learn the noise
%                   for not dummy locations. If noise_struct.learn == 0,
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
function [hypers_learned, Iout, Vout] = gpgrid_dn_sampling(input, Xgrid, noise_struct, xstar, InitParamSet, cov, hyps_in_dim)

% % allocate memory
% Iout = zeros(IoutSize(:)');
% If only one output than only return learned hyperparameters
% Otherwise, also perform prediction for Iout
if(nargout > 1)
    predictionFlag = true;
    if(nargout > 2)
        voutFlag = true;
    else
        voutFlag = false;
    end
else
    predictionFlag = false;
end
% y is a column stack of Isub
y = input.data(:);
% D is the number of dimensions (should be 2 for Images)
D = length(Xgrid);


% prepare a results table with a row for every run, first column will be the 
% nlml value and the rest will be the values of the optimal hyperparameters.
allResultTable = zeros(size(InitParamSet.vals,1),size(InitParamSet.vals,2)+1);


if(~isfield(InitParamSet,'learn') || InitParamSet.learn==true)
%     parfor run_i = 1:size(InitParamSet.vals,1)
     for run_i = 1:size(InitParamSet.vals,1)
                hypers_init = InitParamSet.vals(run_i,:);
                hypers_init = hypers_init(:);
                
%                 % SANITY CHECK - gradient of cov function,
%                 gpr_prodcov_grid_dn_nysapprox - PASSED
%                 checkgrad('gpr_prodcov_grid_dn_nysapprox', hypers_init, 1e-3, Xgrid, input, noise_struct, cov, hyps_in_dim) 
                
                n_hmc_samples = 400;
                hmc_options.num_iters = 1;
                hmc_options.Tau = 20;  % Number of steps.
                hmc_options.epsilon = 0.001;
                hmc_samples = NaN(length(hypers_init),n_hmc_samples);
                hmc_samples(:,1) = hypers_init(:);
                lambda = 1;
                tic
                for n = 2:n_hmc_samples
                    loglikefunc = @(t) gp_grid_prodcov( t, Xgrid, input, noise_struct, cov, hyps_in_dim);
                    logpriorfunc = @(t) gp_grid_Laplace(t, lambda);
                    priorfuns{1}.func = logpriorfunc;
                    priorfuns{1}.indices = 1:floor(length(hmc_samples(:,n - 1))/3);
                    
                    % SANITY CHECK - gradient of cov function,
                    % gp_grid_posterior - PASSED
                    % checkgrad('gp_grid_posterior', hmc_samples(n - 1,:)', 1e-3,loglikefunc,priorfuns)
                    
                    logpostfunc = @(t) gp_grid_posterior(t,loglikefunc,priorfuns);
                    
                    [hmc_samples(:,n), nll, arate, tail{n}] = hmc( logpostfunc, hmc_samples(:,n - 1), hmc_options );
                    n
                end
                toc
                %                 [hypers_learned fx] = minimize2(hypers_init, 'gpr_prodcov_grid_dn_nysapprox', -1000, Xgrid, input, noise_struct, cov, hyps_in_dim);
                disp(hypers_learned);
                allResultTable(run_i,:) = [fx(end),hypers_learned'];
     end
     
    sortResultTable = sortrows(allResultTable,1);
    hypers_learned = sortResultTable(1,2:end)';
    
else
    hypers_learned = InitParamSet.vals(1,:)';
end

% disp(['sf learned = 2^',num2str(imin-1),' ells learned = 2^',num2str(imin_learnedtt(imin)-1)]);
% disp((hypers_learned)./log(2))
% disp(exp(hypers_learned))

% pause
predictionFlag = 0;
if(predictionFlag)
    % call gpr_cov_grid_dn one more time with optimal hyperparameters
    % [nlml, dnlml, alpha_kron, Qs, V_kron] = gpr_cov_grid_dn(hypers_learned, Xgrid, input, noise, cov);
%     [nlml, ~, alpha_kron, Qs, V_kron] = gpr_cov_grid_dn_LRApprox(hypers_learned, Xgrid, input, noise, cov);
    [nlml, ~, alpha_kron, Qs, V_kron] = gpr_prodcov_grid_dn_nysapprox(hypers_learned, Xgrid, input, noise_struct, cov, hyps_in_dim);
    
    %learnexectime =toc
    disp(nlml);
    % if the noise model was spherical and the hyperparamter was learned, then
    % get the new noise matrix
    if(noise_struct.learn == true)
        gamma2 = exp(2*hypers_learned(end));
        noise_struct.var = noise_struct.var*gamma2;
    end
    
    
    % use meshgid to create a locations for interpolation
    % we interpolate also observed locations for denoising
    
    % allocate memory for variance of prediction
    %     Iout = zeros(size(IoutSize));
    %     Vout = zeros(size(IoutSize));
    
    % perform prediction using covMatern covariance function
    if(voutFlag)
        [mu_f, var_f] = gpr_cov_grid_predict_parallel(xstar, hypers_learned, Xgrid, input, alpha_kron, noise_struct, cov, hyps_in_dim, Qs, V_kron);
    else
        [mu_f] = gpr_cov_grid_predict_parallel(xstar, hypers_learned, Xgrid, input, alpha_kron, noise_struct, cov, hyps_in_dim, Qs, V_kron);
        var_f = 0;
    end
    Iout = mu_f;
    Vout = var_f;
    
%     reshape predictions to 2D image form
    %     Iout(sub2ind(size(Iout),xstar(:,2),xstar(:,1))) = mu_f;
    %     Vout(sub2ind(size(Iout),xstar(:,2),xstar(:,1))) = var_f;
    
    
    %     figure(7); imagesc(Iout); drawnow
    %     figure(8); imagesc((Vout)); drawnow
end
% keyboard


end