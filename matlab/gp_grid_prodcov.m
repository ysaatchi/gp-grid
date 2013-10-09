%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% gp_grid_prodcov()
%
% Fast product covariance matrix calculation for GP-grid. If grid is not complete
% or if noise is not homogeneus, use Nystrom approximation. If numeber of
% datapoints is very low, use PHI for better approximation.
%
% Usage: [nlml, dnlml] = gp_grid_prodcov(hypvec, xgrid, input, noise_struct, cov, hyps_in_dim)
%        [nlml, dnlml, alpha_kron, Qs, V_kron] = gp_grid_prodcov(hypvec, xgrid, input, noise_struct, cov, hyps_in_dim)
%
% Inputs:
%       hypvec      hyperparameters vector
%       input       targets corresponding to inputs implied by xgrid
%       	.xgrid       cell array of per dimension grid points
%       noise_struct
%         .learn    flag if to learn noise hyperparameter
%         .var      noise variance of observations. If is a vector and
%                   noise_struct.learn == 1, then will only learn the noise
%                   for not dummy locations. if noise_struct.learn == 0,
%                   then noise_struct.var contains the known noise
%                   variance.
%         .sphericalNoise
%                   learn a single noise hyperparameter for entire input 
%                   space (spherical noise).
%       cov         covariance function as in gpml-matlab
%       hyps_in_dim A cell vector [size D]. Each cell contains a vector of
%                   the hyperparameters in the corresponding dimension. The
%                   parameters must be in the order to be used in the 
%                   covariance function.
%                   
%
% Outputs:
%       nlml        negative log marginal likelihood
%       dnlml       nlml derivative wrt hyperparameters
%       alpha_kron  vector equivalence of the (K^-1)y to use for prediction
%       Qs
%       V_kron
%
%
% SANITY CHECK - gradient of likelihood function,
% gp_grid_prodcov - PASSED
% checkgrad('gp_grid_prodcov', hypers_init, 1e-3, Xgrid, input, noise_struct, cov, hyps_in_dim)
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nlml, dnlml, alpha_kron, Qs, V_kron] = gp_grid_prodcov(hypvec, input, gpmodel)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function parameters
%
% max allowed iteration for PCG
MAXITER = 5000;
% use phi (ratio of sums of eigenvalues) in approximation
USEPHI = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number of elements
N = input.get_N();
% number of real locations
n = input.get_n();
% data in real locations
y = input.zeromeandata(:);
% number of dimensions
D = input.get_D();
% ratio # observed to # of full grid elements
R = n/N;
% dummy noise variance coefficient
% dummyNoise = 1;
% check for valid observation vector
if size(y,1) ~= n
    error('Invalid vector of targets, quitting...');
end
% if length(logtheta) > D+2 || length(logtheta) < D+1
%     error('Error: Number of parameters do not agree with covariance function!')
% end

sphericalNoise = false;
learnNoiseFlag = false;

% if needs to learn a single noise hyperparameter. Can learn a single noise
% hyperparameters for all input locations (aka spherical noise), or a
% single noise parameter to only the locations which are not dummy
% locations.
if( isa(gpmodel.noise_struct,'gp_grid_noise_class') )
    if(gpmodel.noise_struct.learn)
        learnNoiseFlag = true;
        %check if the noise matrix is an eye matrix (spherical noise)
        if(gpmodel.noise_struct.sphericalNoise)
            sphericalNoise = true;
        end
    end
else error('noise_struct should be a struct');
end

% define objects for every dimension
Ks = cell(D,1);
Qs = cell(D,1);
QTs = cell(D,1);
Vs=cell(D,1);
V_kron = 1;
G=zeros(D+1,1);
hlpvars=cell(D,1);
% for calculation of derivatives later on
G(D+1)=1;
% decompose analysis for each dimension
for d = 1:D
    % use input locations from specific dimension
    xg = input.xgrid{d};
    % make sure its a column vector
    xg = xg(:);
 
    % build hyperparameters vector for this dimension 
    hyp_val = hypvec(gpmodel.hyps_in_d{d});
    % calculate covariance matrix using gpml. Since the covariance matrices
    % are for a signal dimension, they will be much smaller than the
    % original covariance matrix
    hlpvar=[];
    if(nargout(gpmodel.cov{1}) == 1)
        [K] = feval(gpmodel.cov{:},hyp_val, xg); %TODO: Toeplitz
    elseif(nargout(gpmodel.cov{1}) == 2)
        [K, hlpvar] = feval(gpmodel.cov{:},hyp_val, xg); %TODO: Toeplitz
    end
    % save covariance matrix for later use
    Ks{d} = 1/2*(K+K');         % avoid numerical errors, force kernel matrix symmetry
    hlpvars{d} = hlpvar;
    if(sum(isnan( Ks{d}(:)))>0)     % check if valid kernel matrix
        nlml = inf;
        dnlml = inf*ones(size(hypvec));
        return;
    end
    % eigendecomposition of covariance matrix
%     try
        [Q,V] = eig(Ks{d}); %TODO: Toeplitz
%     catch exception
%         keyboard;
%     end
    Qs{d} = Q;
    QTs{d} = Q';
%     QTs{d} = inv(Q);
    % make V a column vector
    V = diag(V);
    Vs{d}=V;
    % the eigenvalues of the original covariance matrix are the kron
    % product of the single dimensions eigenvalues
    % this is a vector so still linear in memory
    V_kron = kron(V_kron, V);
    G(d) = length(xg);
end

[V_kron_sort V_index_to_N] = sort(real(V_kron),'descend');

if( sphericalNoise )
    gamma = exp(2*hypvec(end));
    alpha_kron = kron_mv(QTs, y(1:N));
    alpha_kron = alpha_kron./(V_kron + gamma + 1e-8); % jitter for conditioning
    alpha_kron = kron_mv(Qs, alpha_kron);
    
else    % not spherical noise
    
    % if need to learn the noise then use the hyperparameter, noise is
    % approx [eye(n) 0; 0 inf*eye(w)]. We don't really use the dummy noise
    % in the calculation.
    if(learnNoiseFlag)
        gamma = exp(2*hypvec(end));
        noise_struct.var(input.index_to_N) = gamma;
    else
        % calculate the geommetrical mean isotropic noise of the know noise 
        % as an  estimation of the known diagonal noise for log(det()) 
        % calculations. Other estimations can be used although this one 
        % worked best for us
        gamma = real(exp(sum(log(gpmodel.noise_struct.var(input.index_to_N)))/n));
    end
    
    % preconditioning matrix for PCG, only important for diagonal noise
    % when using a known noise model (not learned)
%     C = noise_struct.var(input.index_to_N).^(-0.5);
    C=ones(size(input.index_to_N));
    % maximum iteration should not exceed number of elements
%         max_iter = round(max(N/2,MAXITER));
    max_iter = MAXITER;
    % fast calculation of (K+sn)^-1*y using PCG
    % make threshold lower for better approximation
    [alpha_kron numofitr] = pre_conj_grad_solve(Qs, V_kron, noise_struct.var,...
        input.zeromeandata,input.index_to_N, C,max_iter,1e-2);
    % fprintf('Conjugate gradient solver converged in %i iterations. logtheta =[%d,%d,%d]\n',length(rs),logtheta(1),logtheta(2),logtheta(3));
    
    
    % if number of PCG iteration exceeded max_iter then flag it invalid
    if(numofitr == max_iter)
        nlml = inf;
        dnlml = inf*ones(size(hypvec));
        return;
    end
    
    % SANITY CHECK - for PCG, this is the naive way to calculate alpha_kron
    %     K = kron(Ks{1},kron(Ks{2},Ks{3}));
    %     K1 = K(input.index_to_N,input.index_to_N);
    %     alpha_kron_true = (K1+diag(noise_struct.var(input.index_to_N)))\y;     %%naive alpha calculation
    %     logdet_kron_true = sum(log(eig(K1+diag(noise_struct.var(input.index_to_N))))) %% naive logdet calculation
        
    % can plot of results for testing
    %     G1 = size(Ks{1},1);
    %     G2 = size(Ks{2},1);
    %     G3 = size(Ks{3},1);
    %     a = zeros(G2,G1);
    %     a(input.index_to_N) = real(alpha_kron);
    %     figure(10);imagesc(a(:,:,1)'); xlabel(num2str(logtheta'));colorbar;drawnow
    %     figure(20);plot(phi)
    %         figure(11);imagesc((reshape(real(alpha_kron.*y),G2,G1)))
end

if USEPHI                           % using phi we are able to get better
                                    % results however we are not able to
                                    % justify its use theoretically yet.
    V_N = sum(V_kron_sort); %#ok<UNRCH>
    V_n = sum(V_kron_sort(1:n));
    phi = V_n/V_N;
    % gamma = real(exp(sum(log(noise_struct.var(input.index_to_N)))/n));
    Z = R/phi*V_kron_sort(1:n)+gamma;
    logdet_kron = real(sum(log(Z)));
    dlogdet_dZ=1./Z;       %n vector
    dZ_dphi = -R*V_kron_sort(1:n)/phi^2; %n vector
    dphi_dv = [ones(n,1);zeros(N-n,1)]*1/V_N - V_n/V_N^2; %N vector
    dZ_dV = R/phi*ones(n,1); %n vector
else
    phi = 1;
    % gamma = real(exp(sum(log(noise_struct.var(input.index_to_N)))/n));
    Z = R/phi*V_kron_sort(1:n)+gamma;
    logdet_kron = real(sum(log(Z)));
    dlogdet_dZ=1./Z;       %n vector
    dZ_dphi = 0; %n vector
    dphi_dv = zeros(N,1); %N vector
    dZ_dV = R/phi*ones(n,1); %n vector
end

% calculation of negative log marginal likelihood
% approximated using the logdet approximation
datafit = ((alpha_kron')*y);
complex_penalty = (logdet_kron);
nlml = 0.5*real( datafit+complex_penalty + n*log(2*pi) );


% if(complex_penalty<0)
%     stop;
% end

% approximation without the penalty, good for less smooth
% nlml = real(0.5*((alpha_kron')*y  + n*log(2*pi)));
% disp([num2str(alpha_kron'*y),' ',num2str(logdet_kron),' ',num2str(real(nlml))]);

% if nlml is inf then there is no need to calculate its derivatives
if(nlml == -inf)
    nlml = inf;
    dnlml = inf*ones(size(hypvec));
    return
end

%Now for the derivatives
% P = length(hypvec);
dnlml = zeros(size(hypvec));

% zero pad alpha in dummy locations
alpha_kron_to_N = zeros(N,1);
alpha_kron_to_N(input.index_to_N) = alpha_kron;

% precalculate dV for use later
dVs = cell(D,1);
for d = 1:D
    dVs{d} = diag(QTs{d}*Ks{d}*Qs{d});
end

% tic
% for each dimension

for d = 1:D
    hyps_in_d = gpmodel.hyps_in_d{d};
    hyp_val = hypvec(hyps_in_d);    
    % since the derivative will only effect this dimension, for all other
    % dimensions we can used our stored kernel matrices.
    dK = Ks;
    dV = dVs;
    for hypd_i = 1:length(hyp_val)
        % use saved hlpvar for faster runtime.
        hlpvar=hlpvars{d};
        % use input locations from specific dimension
        xg = input.xgrid{d};
        % make sure its a column vector
        xg = xg(:);
        if(isempty(hlpvar))
            [dK{d}]  = feval(gpmodel.cov{:},hyp_val, xg,[],hypd_i);
        else
            [dK{d}]  = feval(gpmodel.cov{:},hyp_val, xg,[],hypd_i, hlpvar);
        end
        dV{d} = diag(QTs{d}*dK{d}*Qs{d});
        dV_dtheta = 1;
        for d_innerloop = 1:D
            dV_dtheta = kron(dV_dtheta, dV{d_innerloop});
        end
        tr_approx= dlogdet_dZ'*(dZ_dphi*(dphi_dv'*dV_dtheta(V_index_to_N))+dZ_dV.*dV_dtheta(V_index_to_N(1:n)));
        dnlml(hyps_in_d(hypd_i)) = dnlml(hyps_in_d(hypd_i))+ 0.5*(tr_approx - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
    end
end
% toc
% tic
% % TRYING PARALLEL, FOR NOW ITS SLOWER DUE TO OVERHEAD
% % for each dimension
% nhyp = length(hypvec);
% dhypMat = cell(1,D);
% parfor d = 1:D
%     hyps_in_d=[];
%     hyp=[];
%     dnml_local = zeros(nhyp,1);
%     hyps_in_d_mat = sortrows(hyps_in_d{d},1);
%     hyps_in_d.order = hyps_in_d_mat(:,1);
%     hyps_in_d.index = hyps_in_d_mat(:,2);
%     hyps_in_d.val = hypvec(hyps_in_d.index);
%     hyps_in_d.coef = hyps_in_d_mat(:,3);
%     hyp_val = hyps_in_d.coef.*hyps_in_d.val;
%     
%     % since the derivative will only effect this dimension, for all other
%     % dimensions we can used our stored kernel matrices.
%     dK = Ks;
%     dV = dVs;
%     for hypd_i = hyps_in_d.order(:)'
%          if(isempty(hlpvars))
%             [dK{d}]  = feval(cov{:},hyp_val, xg,[],hypd_i);
%         else
%             [dK{d}]  = feval(cov{:},hyp_val, xg,[],hypd_i, hlpvars);
%         end
%         dV{d} = diag(QTs{d}*dK{d}*Qs{d});
%         dV_dtheta = 1;
%         for d_innerloop = 1:D
%             dV_dtheta = kron(dV_dtheta, dV{d_innerloop});
%         end
%         tr_approx= dlogdet_dZ'*(dZ_dphi*(dphi_dv'*dV_dtheta(V_index_to_N))+dZ_dV.*dV_dtheta(V_index_to_N(1:n)));
% %         dnlml(hyps_in_d.index(hypd_i)) = dnlml(hyps_in_d.index(hypd_i))+ hyps_in_d.coef(hypd_i) * 0.5*(tr_approx - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
%          dnml_local(hyps_in_d.index(hypd_i))= hyps_in_d.coef(hypd_i) * 0.5*(tr_approx - alpha_kron_to_N'*kron_mv(dK, alpha_kron_to_N));
%     end
%     dhypMat{d} = dnml_local;
% end
% dnlml = sum(cell2mat(dhypMat),2);
% toc


% noise
if(learnNoiseFlag)
    dnlml(end) = 0.5*(sum(dlogdet_dZ) - alpha_kron'*alpha_kron)*2*gamma; % Adjust for calculating derivatives in log space  
end

% just a test need to take off
dnlml = real(dnlml);
% dnlml = real(dnlml);
%  disp(logtheta');
% disp([nlml,datafit,complex_penalty,dnlml']);

% fftinput.xgrid{1} = -max(input.xgrid{1}):0.1:max(input.xgrid{1});
% fftinput.xgrid{2} = -max(input.xgrid{2}):0.1:max(input.xgrid{2});



% fftinput.xgrid{1} = -.50:0.01:.50;
% fftinput.xgrid{2} = -.50:0.01:.50;
% fftinput.xgrid = input.xgrid;

% specdensityImage = multdimSpecdensity(fftinput.xgrid,hypvec(:),hyps_in_d);
% figure(10);imagesc((specdensityImage'));%colormap(gray);
% drawnow;

%  stop = 1;

% % test here for shrinking wights
% dnlml(1:((length(hypvec)-1)/3)) = dnlml(1:num_of_w) + lambda*exp(hypvec(1:num_of_w));


