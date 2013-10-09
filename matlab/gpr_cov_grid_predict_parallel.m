%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Elad Gilboa
% 2013
%
% gpr_cov_grid_predict_parallel()
%
% Fast perallel prediction for GP-grid by using alpha which is the vector
% equivalence of (K^-1)y calculated by gpr_cov_grid_dn
%
% Usage: [mu_f, std_f] = gpr_cov_grid_predict_multi(xstars, logtheta, xgrid, ygrid, alpha, cov)
%        [mu_f] = gpr_cov_grid_predict_multi(xstars, logtheta, xgrid, ygrid, alpha, cov)
%
% Inputs:
%       xstars      locations for prediction
%       logtheta    hyperparameters for covariance function
%       xgrid       cell array of per dimension grid points
%       ygrid       targets corresponding to inputs implied by xgrid
%       alpha       vector equivalence of (K^-1)y
%       cov         covariance function as in gpml-matlab
%
% Outputs:
%       mu_f        posterior mean for xstars locations
%       var_f       posterior variance for xstars locations (not
%                   implemented)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mu_f, var_f] = gpr_cov_grid_predict_parallel(xstars, hypvec, input, gpmodel, alpha, V_kron, Qs)

tic
%%%
% function parameters
%
% max allowed iteration for PCG
MAXITER = 10000;
%
%%%

% if number of out arguments is 1 then only calculate posterior mean (much
% faster)
if(nargout > 1)
    std_flag = true;
else
    std_flag = false;
end

% number of elements
N = input.get_N();
% number of real locations
%n = length(input.index_to_N);
% data in real locations
%y = input.data(:);
% number of dimensions
D = input.get_D();
% check for valid observation vector
assert(D > 0);
assert(size(xstars,2) == D);
% assert(size(ygrid,1) == N);
% number of prediction locations
M = size(xstars,1);

mu_f = zeros(M,1);

% DummyLocations = logical(abs(alpha) < 0.01);

% since prediction calculations are usually very repetative we precalculate
% Kxy covariance matrices using only unique values for each dimension in
% xstarts
Kxy_sparse = cell(D,2);
Kxtxt = 1;
for d=1:D
    xg = input.xgrid{d};
    % make sure its a column vector
    if size(xg,2) > size(xg,1)
        xg = xg';
    end
    [b, ~, n] = unique(xstars(:,d));
%     hyp_cov = [logtheta(d); logtheta(D+1)/D];
    hyps_in_d = gpmodel.hyps_in_d{d};
    hyp_val = hypvec(hyps_in_d);
    
    Kxy_sparse{d,1} = feval(gpmodel.cov{:},hyp_val, xg, b);
    % save the reference to all xstars with this input in this dimension
    Kxy_sparse{d,2} = n;
    if(std_flag)
        Kxtxt = Kxtxt*(feval(gpmodel.cov{:},hyp_val, 1));       % calculate variance of a single location
    end
end

toc

var_f = zeros(M,1);
% preconditioning matrix for PCG
C = gpmodel.noise_struct.var(input.index_to_N).^(-0.5);
% maximum iteration should not exceed number of elements
max_iter = min(N,MAXITER);

index_to_N = input.index_to_N;
% tic
for m = 1:M
    
    Kstar_kron = 1;
    for d = 1:D
        % rebuild full matrix from sparse representation
        Kxy_d = Kxy_sparse{d,1}(:,Kxy_sparse{d,2}(m));
        % Kstar_kron = Kxx*
        Kstar_kron = kron(Kstar_kron, Kxy_d);
    end
%     toc
%     tic
    % calculate GP posterior mean Kxy*alpha = Kxy*(K+sn)^-1*y
    alpha_to_N = zeros(N,1);
    alpha_to_N(index_to_N)=alpha;
    mu_f(m) = real(Kstar_kron'*alpha_to_N);
    
%     if(std_flag)
%         % make sure Kstar_kron a column vector
%         if size(Kstar_kron,2) > size(Kstar_kron,1)
%             Kstar_kron = Kstar_kron';
%         end
%         % fast calculation of alpha_kron = (Kxx+sn^2)^-1*Kxx* using PCG
%         %[alpha_kron rs] = pre_conj_grad_solve(Qs, V_kron, noise, Kstar_kron_input, C, max_iter);
%         [alpha_kron rs] = pre_conj_grad_solve_wrapper(Qs, V_kron, noise, Kstar_kron, C, max_iter,index_to_N);
%         alpha_to_N = zeros(N,1);
%         alpha_to_N(index_to_N)=alpha_kron;
%         % calculate posterior variance = Kx*x* - Kx*x (Kxx-sn^2)^-1 Kxx*
%         var_f(m) = Kxtxt - Kstar_kron'*alpha_to_N;
%         
%     end
    
%     disp(['m=',num2str(m),' ',num2str(exctime)])
end
% m_exctime = toc
if(std_flag)
%     tic
    parfor m = 1:M
        %     exctime_start = toc;
        Kstar_kron = 1;
        for d = 1:D
            % rebuild full matrix from sparse representation
            Kxy_d = Kxy_sparse{d,1}(:,Kxy_sparse{d,2}(m));
            % Kstar_kron = Kxx*
            Kstar_kron = kron(Kstar_kron, Kxy_d);
        end
        %     toc
        %     tic
        % calculate GP posterior mean Kxy*alpha = Kxy*(K+sn)^-1*y
        %     alpha_to_N = zeros(N,1);
        %     alpha_to_N(index_to_N)=alpha;
        %     mu_f(m) = real(Kstar_kron'*alpha_to_N);
        
        if(std_flag)
            % make sure Kstar_kron a column vector
            if size(Kstar_kron,2) > size(Kstar_kron,1)
                Kstar_kron = Kstar_kron';
            end
            % fast calculation of alpha_kron = (Kxx+sn^2)^-1*Kxx* using PCG
            %[alpha_kron rs] = pre_conj_grad_solve(Qs, V_kron, noise, Kstar_kron_input, C, max_iter);
            [alpha_kron, ~] = pre_conj_grad_solve(Qs, V_kron, gpmodel.noise_struct.var, Kstar_kron(index_to_N),index_to_N, C, max_iter,1e-3);
            alpha_to_N = zeros(N,1);
            alpha_to_N(index_to_N)=alpha_kron;
            % calculate posterior variance = Kx*x* - Kx*x (Kxx-sn^2)^-1 Kxx*
            var_f(m) = Kxtxt - Kstar_kron'*alpha_to_N;
            
        end
        %     exctime = toc
        %     disp(['m=',num2str(m),' ',num2str(exctime-exctime_start)])
        disp(['m=',num2str(m)]);
        
    end
%     v_exctime = toc
end

% save('predict_times','m_exctime','v_exctime');


