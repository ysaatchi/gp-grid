function [Xfull, logtheta_learned] = gpgrid_img(Isub,Xgrid, ver_size, hor_size, suglogtheta, learnflag,noise)

% [xsize,ysize] = size(Isub);

% [X1,X2] = meshgrid(1:upsamp:upsamp*xsize,1:upsamp:ysize);

if(nargout > 1) 
%     prediction = false;
prediction = true;
else
    prediction = true;
end
y = Isub(:);
D = length(Xgrid);

if(nargin<5)
    suglogthetaFlag=false;
    s0 = var(y)+1e-5;
    sf =ones(D,1)*sqrt(s0/4); 
    sn = sqrt(s0)/100;
else
     suglogthetaFlag=true;
     if(suglogtheta(1) == -inf)
         suglogthetaFlag=false;
     end
     sf =exp(suglogtheta(D+1)); 
     sn = exp(suglogtheta(D+2));
end

if(nargin <6)
    learnflag = true;
end

% x = [X1(:) X2(:)];

% Y = Isub;

%[N,D] = size(x);

[hor_starvec,ver_starvec] = meshgrid(1:1:ver_size,1:1:hor_size);
xstar = [ver_starvec(:),hor_starvec(:)];

% cov_func = 'gpr_covMaterniso_grid';

if(suglogthetaFlag == false)
    %     % ellscoef = [1 5 10 20 50 100];
    %     ellscoef = 2.^(0:10);
    %     parfor ellsi = 1:length(ellscoef)
    %         ells = ellscoef(ellsi)*ones(1,D);
    
    ellscoef = 2.^(-2:3:10);
    sfcoef = 2.^[1,5];
    % ellscoef = 2.^5;
    for sfi = 1:length(sfcoef)
        parfor ellsi = 1:length(ellscoef)
            ells = ellscoef(ellsi)*ones(D,1);
            
            %     logtheta_init = log([ells'; ones(D,1)*sqrt(s0); 0]);
            logtheta_init = log([ells; sfcoef(sfi);sn]);
            
            
            %tic
            %     [logtheta_learned fx] = minimize([logtheta_init(1:D); logtheta_init(D+1); logtheta_init(end)], 'gpr_covSEard_grid', 100, Xgrid, y);
            [logtheta_learned fx] = minimize([logtheta_init(1:D); logtheta_init(D+1); logtheta_init(end)], 'gpr_covMaterniso_grid', 100, Xgrid, y);
            
            fxt(ellsi) = fx(end);
            logtheta_learnedt(:,ellsi) = logtheta_learned;
            %
        end
        [fmin imin] = min(fxt);
        logtheta_learnedtt(:,sfi) = logtheta_learnedt(:,imin);
        fmin_learnedtt(sfi) = fmin;
        imin_learnedtt(sfi) = imin;
    end
    [fmin imin] = min(fmin_learnedtt);
    logtheta_learned = logtheta_learnedtt(:,imin);
else
    if(learnflag == true)
        [logtheta_learned fx] = minimize(suglogtheta, 'gpr_covMaterniso_grid', 100, Xgrid, y);
%         [logtheta_learned fx] = minimize(suglogtheta, 'gpr_covMaterniso_grid', 100, Xgrid, y);
    else
        logtheta_learned = suglogtheta;
    end
end

% [nlml, dnlml, alpha_kron] = gpr_covSEard_grid(logtheta_learned, Xgrid, y);
[nlml, dnlml, alpha_kron] = gpr_covMaterniso_grid(logtheta_learned, Xgrid, y);
%learnexectime =toc
tic
% [i,j] = ind2sub([xsize,ysize],XtestInd)
% [i,j] = ind2sub([xsize,ysize],1:prod([xsize,ysize]));


% Xfull = zeros(xsize,ysize);
% if(prediction)
%     parfor i = 1:xsize*ysize
%         [f, df] = gpr_covMaterniso_grid_predict(xstar(i,:)', logtheta_learned, Xgrid, y, alpha_kron);
%         Xfull(i) = f;
%     end
%     predictexectime = toc
%     figure(6); imagesc(Xfull); drawnow
% end
Xfull = zeros(ver_size,hor_size);
if(prediction)
    [mu_f, std_f] = gpr_covMaterniso_grid_predict_multi(xstar, logtheta_learned, Xgrid, y, alpha_kron);
    Xfull(sub2ind(size(Xfull),xstar(:,2),xstar(:,1))) = mu_f;
    figure(7); imagesc(Xfull); drawnow
end

end
