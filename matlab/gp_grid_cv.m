function [hypers_learned nlml] = gp_grid_cv(nfolds, input, gpmodel, xstar, vset)
%
% Perform cross validation for gp_grid.
%
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = length(input.xgrid);
% randcvindx = cell(D,1);
Gs = gp_grid_size(input.xgrid);
nlml = inf;


%%% TEST
numberstarts = 1;


cvsets = gp_grid_cv_makecvsets(input, nfolds, 'sets',vset,'balance',true);
% cvsets = gp_grid_cv_makecvsets(input.xgrid, input, nfolds, 'sets','rand','balance',true);
% cvsets = gp_grid_cv_makecvsets(input.xgrid, input, nfolds, 'sets','rand','balance',false);
% cvsets = gp_grid_cv_makecvsets(input.xgrid, input, nfolds, 'sets','voroni','balance',false);
% cvsets = gp_grid_cv_makecvsets(input.xgrid, input, nfolds, 'sets','voroni','balance',true);

%     % build a data matrix
%     trueData = NaN(Gs);
%     trueData(input.index_to_N) = input.data;

%%% TEST!!!!
% load('voroni_cvsets');
%%%%


lambda_vals = [0 1e-2 1  100 1000];
L = length(lambda_vals);
resultssmse = zeros(L,2);
resultsnlml = zeros(L,2);

% for numoftries = 1:4    % give a few tries just in case the initial values are very bad
%     Fs = 100;       %need to fix
%     InitParamSet.learn = true;
%     InitParamSet.vals = [make_gpsmp_hyps_initvals(hyps_in_dim,(std(input.data)/length(hyps_in_dim{1}))^(1/D),0,0,Fs/6,2*Gs(1)/Fs,0,Fs,2)',exp(0.2*rand)];
% %     InitParamSet.vals = [make_gpsmp_hyps_initvals(hyps_in_dim,(std(input.data)/length(hyps_in_dim{1}))^(1/D),0,0,2,-3,1,Fs)',exp(0.2*rand)];
%     [hypers_learned, trnlml] = gpgrid(input, input.xgrid, noise_struct, xstar, InitParamSet, cov, hyps_in_dim);
%     if(trnlml.numofitr>30)    % valid optimization if num of iterations excided 5% of allowed iterations
%         break;
%     end
% end

% %%%%% TEST!!!!!!!!
% load('C:\Users\Elad\Dropbox\gp\Pattern Discovery\results\tempInit')
% InitParamSet.learn = true;
% InitParamSet.vals = hypers_init_cells{1,1};
% %%%%%%%%%%%%

    smsevec = zeros(nfolds,L);
    nlmlvec = zeros(nfolds,L);
    teststruct(L,nfolds) = struct('hypers_learned',[],'tr',[],'tst',[],'Iout',[],'Gs',[]); %cell(nfolds,L);
    
for lam_i = 1:L
    lambda = lambda_vals(lam_i);
    for cvtestnum = 1:nfolds
        
%         [tr, tst, xstar] = gp_grid_set_splitsets(cvsets{cvtestnum}, input, gpmodel);
        [tr.input, tst.input, xstar] = input.splitsets(cvsets{cvtestnum});
        
        %         %%%% TEST for single cv set!!!!
        %         [trinput.xgrid, cvinput.xgrid, trInput, cvInput, trnoise_struct, cvnoise_struct, cvxstar, cvxstar_sub] = gp_grid_cv_splitsets(cvsets, 1, input, input.xgrid, noise_struct);
        %         %%%%%%%%%%%%%%
        
          
        gpmodel.logpriorfuns{1}.func = @(t) gp_grid_Laplace(t, lambda);
        gpmodel.logpriorfuns{1}.indices = 1:floor(gp_grid_numofhyps(gpmodel.hyps_in_d)/3);
        
        
        %         for numoftries = 1:4    % give a few tries just if initial values are very bad and can't converge
        %             tr.InitParamSet.learn = true;
        %             %             InitParamSet.vals = [make_gpsmp_hyps_initvals(hyps_in_dim,0,0,-1,2,log((1/70)),0,Fs),exp(0.2*rand)];
        %             %             InitParamSet.vals = [make_gpsmp_hyps_initvals(hyps_in_dim,(std(input.data)/length(hyps_in_dim{1}))^(1/D),0,1,Fs,2*Gs(1)/Fs,0,Fs,2)',exp(0.2*rand)];
        %             params.Fs = Fs;
        %             params.wm = std(input.data);
        %             params.sm = 2*Gs/Fs;
        %             tr.InitParamSet.vals = [make_gpsmp_hyps_initvals(input.xgrid, input, hyps_in_dim,3,params)', -1];
        %
        %             [hypers_learned, trnlml, Iout] = gpgrid(tr.input, tr.input.xgrid, tr.noise_struct, cvxstar, tr.InitParamSet, cov, hyps_in_dim, logpriorfuns);
        %             if(trnlml.numofitr>30)    % valid optimization if num of iterations excided 5% of allowed iterations
        %                 break;
        %             end
        %         end
        %
        %
        %         msevec(cvtestnum) = mean((Iout-cv.input.data).^2);
        %         smsevec(cvtestnum,lam_i) = mean((Iout-cv.input.data).^2)/var(cvInput.data);
        %         nmsevec(cvtestnum) = mean((Iout-cv.input.data).^2./cvInput.data.^2);
        %
        %         cv.InitParamSet.vals = hypers_learned';
        %         cv.InitParamSet.learn = false;
        %         [~, cvnlml] = gpgrid(cv.input, cv.input.xgrid, cv.noise_struct, [], cv.InitParamSet, cov, hyps_in_dim, []);
        %         nlmlvec(cvtestnum,lam_i) = cvnlml.nlml;
        %         hypers_learned_cells{lam_i,cvtestnum} = hypers_learned;
        %         hypers_init_cells{lam_i,cvtestnum} = InitParamSet.vals;
        
        
        
        tr.gpmodel.learn = true;
        params.wm = std(tr.input.get_data());
        params.sm = 2*Gs./input.Fs;
        tr.gpmodel.hyperparams = zeros(numberstarts,sum(gp_grid_numofhyps(tr.gpmodel.hyps_in_d))+1);
        
        for numoftries = 1:4    % give a few tries just if initial values are very bad and can't converge
            for nrandstart=1:numberstarts
                tr.gpmodel.hyperparams(nrandstart,:) = [make_gpsmp_hyps_initvals(tr.input, tr.gpmodel, 3, params)', -1];
            end
            [hypers_learned, trnlml, Iout] = gpgrid(tr.input, tr.gpmodel, xstar,10);
            
            if(trnlml.numofitr>20*0)    % valid optimization if num of iterations > min thresh
                break;
            end
        end
        
       smsevec(cvtestnum,lam_i) = mean((Iout-tst.input.zeromeandata).^2)/var(input.get_data());
        
        temp_input = gp_grid_input_class(tst.input);
        tst.gpmodel.hyperparams = hypers_learned';
        tst.gpmodel.learn = false;
        temp_input.zeromeandata = tst.input.get_data()  - tr.input.meandata;
        [~, tstnlml] = gpgrid(temp_input, tst.gpmodel, []);
        
        nlmlvec(cvtestnum,lam_i) = tstnlml.nlml;
        teststruct(lam_i,cvtestnum).hypers_learned = hypers_learned;
        teststruct(lam_i,cvtestnum).tr = tr;
        teststruct(lam_i,cvtestnum).tst = tst;
        teststruct(lam_i,cvtestnum).Iout = Iout;
        teststruct(lam_i,cvtestnum).Gs = Gs;

        save('C:\Users\Elad\Dropbox\gp\Pattern Discovery\results\cv_run')
    end
    resultssmse(lam_i,:) = [mean(smsevec(:,lam_i)) std(smsevec(:,lam_i))];
    resultsnlml(lam_i,:) = [mean(nlmlvec(:,lam_i)) std(nlmlvec(:,lam_i))];
    save('C:\Users\Elad\Dropbox\gp\Pattern Discovery\results\cv_run')
end


end

function cvsets = gp_grid_cv_makecvsets(input, nsets, varargin)

pnames = {   'sets' 'balance'};
dflts =  {'voroni' true};
[sets balance] ...
    = internal.stats.parseArgs(pnames, dflts, varargin{:});

all_subscripts = makePossibleComb(input.xgrid);
unmasked_subs = all_subscripts(input.index_to_N,:);
n = length(input.index_to_N);
randperm_indx = randperm(n);
randperminput = input.index_to_N(randperm_indx);
cvsets = cell(nsets,1);
switch sets
    case 'rand'
        if(~balance)
            options = statset('MaxIter',1);
        else
            options = statset();
        end
        IDX = kmeans(1:n,nsets,'start','uniform','options',options);
        for i = 1:max(IDX)
            cvsets{i} = randperminput(IDX==i);
        end
    case 'voroni'
        centers = unmasked_subs(randperm_indx(1:nsets),:);
        if(~balance)
            options = statset('MaxIter',1);
        else
            options = statset();
        end
        IDX = kmeans(unmasked_subs,nsets,'start',centers,'options',options);
        for i = 1:max(IDX)
            cvsets{i} = input.index_to_N(IDX==i);
        end
end

tmpData = zeros(length(input.xgrid{2}),length(input.xgrid{1}));
for i = 1:nsets
    tmpData(cvsets{i}) = i;
end
figure; imagesc(tmpData)

end

