%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Elad Gilboa
% 2013
%
% gpr_cov_grid_predict_parallel()
%
% Fast covariance matrix calculation for GP-grid
%
% Usage: [mu_f, std_f] = gpr_cov_grid_predict_multi(xstars, logtheta, xgrid, ygrid, alpha, cov)
%
% Inputs: 
%       logtheta    hyperparameters vector
%       Xgrid       grid input locations of observations
%       y           observations vector
%       noise       noise variance of observation
%       cov         covariance function as in gpml-matlab
% 
% Outputs: 
%       nlml        negative log marginal likelihood            
%       dnlml       nlml derivative wrt hyperparameters
%       alpha_kron  vector equivalence of the (K^-1)y to use for prediction
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mu_f, std_f] = gpr_cov_grid_predict_parallel(xstars, logtheta, xgrid, ygrid, alpha, cov)

%Fast version of GP prediction ALL points lie on a grid (with G grid points per dim)
%This version also attempts to do everything in linear memory complexity
%NOTE: Only works with axis-aligned covariance functions (e.g. covSEard) + spherical noise (e.g. covNoise)

%xgrid : cell array of per dimension grid points
%ygrid : targets corresponding to inputs implied by xgrid
%xstar : new input point where we want to make predictions

D = length(xgrid); %number of dimensions
N = prod(cellfun(@length, xgrid));
assert(D > 0);
assert(size(xstars,2) == D);
assert(size(ygrid,1) == N);
M = size(xstars,1);

%tic;
Kstars = cell(D,1);
Qs = cell(D,1);
QTs = cell(D,1);
Vs = cell(D,1);
mu_f = zeros(M,1);
std_f = zeros(M,1);
for d = 1:D
    xg = xgrid{d}'; 
    hyp.cov = [logtheta(d); logtheta(D+1)/D];
    K_d = feval(cov{:}, hyp.cov, xg);
    [Q,V] = eig(K_d);
    K_ds{d} = K_d;
    Qs{d} = Q;
    QTs{d} = Q';
    Vs{d} = V;
end


Kxy = cell(D,2);            %precalculate all covariance calculations, since these are usually very repetative
for d=1:D
    xg = xgrid{d}';
    [b, m, n] = unique(xstars(:,d));        
    hyp_cov = [logtheta(d); logtheta(D+1)/D];
    Kxy{d,1} = feval(cov{:},hyp_cov, xg, b);
    Kxy{d,2} = n;
end



% Kstars = cell(D,1);
parfor m = 1:M
    Kstar_kron = 1;
    beta_kron = 1;
    Kstar_kron1=1;
    V_kron = 1;
    for d = 1:D
        xg = xgrid{d}';
        hyp_cov = [logtheta(d); logtheta(D+1)/D];
        %     K_d = feval('covMaterniso', 5,hyp.cov, xg);
%         [Kstar_d] = feval('covMaterniso', 5,hyp_cov, xg, xstars(m,d));
        [Kstar_d] = Kxy{d,1}(:,Kxy{d,2}(m));
        %     [Q,V] = eig(K_d{d});
%         Q = Qs{d};
%         V = Vs{d};
%         beta = Q'*Kstar_d;
%         V = diag(V);

%%%% dont need since initialized kron to 1;
%         if d == 1
% %             V_kron = V;
% %             beta_kron = beta;
%             Kstar_kron = Kstar_d;
%         else
%%%%% 

%             V_kron = kron(V_kron, V); %this is a vector so still linear in memory
%             beta_kron = kron(beta_kron, beta);


            Kstar_kron = kron(Kstar_kron, Kstar_d);
%              Kstar_kron = kron(Kstar_d,Kstar_kron);
             
             
% %         end
        % do not need to store here, but useful for debugging
%         Kstars{d} = Kstar_d;
        %     Qs{d} = Q;
        %     QTs{d} = Q';
        
    end
    % noise_var = exp(logtheta(D+2))^2;
    % V_kron = V_kron + noise_var*ones(N,1);
    %printf('Inverse condition number = %e\n', min(V_kron)/max(V_kron));
    
%     b = reshape(Kstar_kron,55,60)';
%     Kstar_kron = b(:);
        
    mu_f(m) = real(Kstar_kron'*alpha);
%     figure(15); imagesc(reshape(Kstar_kron.*alpha,60,75));
%     drawnow;
%     xstars(m,:)
    
    % took off to save computational time, dont need it now
    % Kss = feval('covMaterniso', 5,hyp.cov,  xstar');
    % std_f = sqrt(Kss - sum((beta_kron.^2)./V_kron));
    std_f(m) = 0;
end


%toc
    

