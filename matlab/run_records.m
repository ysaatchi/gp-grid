clear;
close all;

load ~/PhD/datasets/classification/records.mat;

[N,D] = size(Xall);
Ntrain = round(N*0.8);
Ntest = N - Ntrain;

[rr, ri] = sort(rand(N,1));
Xtrain = Xall(ri(1:Ntrain), :);
Xtest = Xall(ri(Ntrain+1:end), :);
ytrain = yall(ri(1:Ntrain));
ytest = yall(ri(Ntrain+1:end));

Xgrid = cell(D,1);
for d = 1:D
    ud = unique(Xall(:,d));
    Xgrid{d} = ud;
end

clear Xall; clear yall;

%%% k-NN using kd-tree
% Build the k-d Tree once from the training data.
tic;
[tmp, tmp, TreeRoot] = kdtree( Xtrain(1:1000000, :), [] );
%% Find the closest point row index in Xtrain for each point in Xtest
[ ClosestPtIndex, DistB, TreeRoot ] = kdtreeidx([], Xtest, TreeRoot);
ytr = ytrain(1:1000000); 
ypred = ytr(ClosestPtIndex);
test_error_knn = 1 - sum(ypred==ytest)/size(ytest, 1);
%%% Free the k-D Tree from memory.
kdtree([],[],TreeRoot);
results.nn_runtime = toc;
results.test_error_nn = test_error_knn;

tic;
pc = kron_hist(Xtrain(ytrain == 1, :), Xgrid) + 1;
nc = kron_hist(Xtrain(ytrain == -1, :), Xgrid) + 1;
hist_runtime = toc;
results.hist_runtime = hist_runtime;

%X = cartprod(Xgrid);
y = pc ./ (pc + nc);
noise_var = (pc .* nc) ./ (((pc + nc).^2) .* (pc + nc + 1)); 

pcounts = kron_hist(Xtest(ytest == 1, :), Xgrid);
ncounts = kron_hist(Xtest(ytest == -1, :), Xgrid);

%%% HISTOGRAMMING
mu_hist = pc ./ (pc + nc);
test_error_hist = 1 - (sum(pcounts .* (mu_hist >= 0.5)) + sum(ncounts .* (mu_hist < 0.5)))/(sum(pcounts) + sum(ncounts));
results.test_error_hist = test_error_hist;

%%% KRONECKER
tic;
mu = gpr_covSEard_grid_dn_predict([-2*ones(D,1); 0], Xgrid, y, noise_var);
test_error_grid = 1 - (sum(pcounts .* (mu >= 0.5)) + sum(ncounts .* (mu < 0.5)))/(sum(pcounts) + sum(ncounts));
grid_runtime = toc;
results.test_error_grid = test_error_grid;
results.grid_runtime = grid_runtime;

%%% OPTIMIZATION CODE
% log_ells = (-5:1:1);
% log_sigfs = (-3:1:1);
% for i = 1:length(log_ells)
%     for j = 1:length(log_sigfs)
%         nlml(i,j) = gpr_covSEard_grid_dn([log_ells(i)*ones(D,1); log_sigfs(j)], Xgrid, y, noise_var);
%     end
% end
% if length(y) < 5000
%     [nlml2, dnlml2, alpha2, logdet2] = gpr_covSEard_dn(ones(D+1,1), Xgrid, y, noise_var);
% end

%%% LINEAR LOGISTIC
tic;
ytr = ytrain;
ytr(ytr == -1) = 0;
mu_lin = linear_logistic(Xtrain, ytr, Xtest, 20);
ypred = -ones(size(ytest));
ypred(mu_lin >= 0.5) = 1;
test_error_lin = 1 - sum(ypred==ytest)/size(ytest, 1);
nll_lin = -mean(ytest.*log(mu_lin) + (1-ytest).*log(1-mu_lin));
lin_runtime = toc;
results.test_error_lin = test_error_lin;
results.lin_runtime = lin_runtime;

save(sprintf('~/PhD/src/matlab/gpr_grid/run_%s_%s.mat', 'records', datestr(now, 'yyyymmdd')), 'results');




