% clear;
close all;

subSampling = 1;
xsize=50;
ysize=50;
tframe=1;

rgb = imread('gpml.png');
im = mean(rgb,3);
vid = im(200:200+xsize,200:200+ysize);

[X1,X2] = meshgrid(1:subSampling:xsize,1:subSampling:ysize);


x = [X1(:) X2(:)];
y = vid(sub2ind(size(vid),x(:,1),x(:,2),tframe*ones(size(x,1),1)));
Y = reshape(y,size(X1));

[N,D] = size(x);


ells = [1 1];
s0 = var(y)/10;
logtheta_init = log([ells'; ones(D,1)*sqrt(s0/D); 2]);

[xstarvec,ystarvec] = meshgrid(1:1:xsize,1:1:ysize);
xstar = [xstarvec(:),ystarvec(:)];
M = size(xstar,1);


%% Kronecker

Xgrid = cell(D,1);
for d = 1:D
    Xgrid{d} = 1:subSampling:xsize;
end

% Xgrid{3} = 10;
% likfunc = @likGauss;
% covfunc = @gpr_covSEard_grid; hyp.cov = log(2.2*rand(D,1)); hyp.lik = log(0.1);
% %disp('hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y)');
% hyp = minimize(hyp, @gp, -100, @infExact, [], covfunc, likfunc, x, y);
% disp(' ');

% logtheta = log([2.2*rand(D,1); 1;0.1]);
tic
logtheta_learned = minimize([logtheta_init(1:D); log(sqrt(s0)); logtheta_init(end)], 'gpr_covSEard_grid', -100, Xgrid, y);
%


[nlml, dnlml, alpha_kron] = gpr_covSEard_grid(logtheta_learned, Xgrid, y);
learnexectime =toc
tic
% [i,j] = ind2sub([xsize,ysize],XtestInd)
% [i,j] = ind2sub([xsize,ysize],1:prod([xsize,ysize]));

fbar = zeros(xsize,ysize);
parfor i = 1:xsize*ysize
    [f, df] = gpr_covSEard_grid_predict(xstar(i,:)', logtheta_learned, Xgrid, y, alpha_kron);
    fbar(i) = f;
    i
end
predictexectime = toc
% save('gpr_grid_results')
close all;
% fig1 = figure;
% axes1 = axes('Parent',fig1);
% % view(axes1,[-26 34]);
% hold(axes1,'all');
% h = surface(xstarvec(1,:),ystarvec(:,1),real(fbar))
% axis([1 40 1 40 1, 200])
% return;
% figure; surface(xstarvec(1,:),ystarvec(:,1),vid(100:-1:1,1:100))
% figure; surface(X1(1,:),X2(:,1),vid(100:-2:1,1:2:100));

figure; imshow(vid,[0,255])
figure; imshow(Y',[0,255])
figure; imshow(real(fbar'),[0,255])
% for i=1:ceil(size(xstar,1)/100)
%     startIdx = (i-1)*100+1;
%     endIdx = startIdx + min(100,size(xstar,1)-startIdx) - 1;
%     [f, df] = gpr_covSEard_grid_predict(xstar(startIdx:endIdx,:), logtheta_learned, Xgrid, y, [], [],locate_info,0);
%     if(length(f)<100)
%         f(100)=0;
%     end
%     fbar(:,i) = f;
% end
%
% fbar = f;
% vid(xstar(1),xstar(2),1)

return;


%% Full GP (tensor)


tic;

%        y((ysize/2)*(xsize/4)+(xsize/4)) = 30;




% covfunc = @covSEiso;
% likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
% hyp2.cov = [0 ; 0];
% hyp2.lik = log(0.1);

covfunc = @covSEiso;
likfunc = @likGauss; sn = 0.1; hyp.lik = log(logtheta_learned(4));
hyp2.cov = [logtheta_learned(1) ; logtheta_learned(2)];
hyp2.lik = log(logtheta_learned(3));


hyp2 = minimize(hyp2, @gp, -50, @infExact, [], covfunc, likfunc, x, y);
%%
[mus vars] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, xstar);

%%
%     figure;
%     imshow(reshape(mus,size(Y)),[0,255])

figure
z = (1:xsize)';
mus1 = reshape(mus,size(xstarvec));
vars1 = reshape(vars,size(xstarvec));
mus1 = mus1((xsize/2),:)';
vars1 = vars1((xsize/2),:)';
y1 = reshape(y,size(X1));
f = [mus1 + sqrt(vars1);flipdim(mus1 - sqrt(vars1),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on
plot(1:2:xsize,y1((xsize/2),:),'--r')
hold on
plot(1:xsize,mus1)
%     plot(1:20,vid(1:20,1,3),'--g')
hold off
legend('confidence','frame(2)','GP interpolation','frame(3)')

nlml2 = arrayfun(@(i)gp(hyp2, @infExact, [], covfunc, likfunc, x(i,:), y(i)),(1:length(y)));
nlml2 = reshape(nlml2, size(Y));

figure
surface(xstarvec(1,:),ystarvec(:,1),reshape(mus,size(xstarvec)))
figure
surface(xstarvec(1,:),ystarvec(:,1),reshape(sqrt(vars),size(xstarvec)))

figure
surface(xstarvec(1,:),ystarvec(:,1),reshape(mus,size(xstarvec)),zeros(size(xstarvec)))
hold on
surface(X1(1,:),X2(:,1),Y,nlml2)
hold off


figure
surface(X1(1,:),X2(:,1),reshape(nlml2,size(X1)))

return
mnlp_test = 0.5*(log(2*pi) + mean(log(vars)) + mean(((ystar - mus).^2)./vars));
nmse_test = mean(((ystar - mus).^2))/mean(ystar.^2);
fprintf('Testing NMSE (joint GP) = %3.5f\n', nmse_test);
fprintf('Testing MNLP (joint GP) = %3.5f\n', mnlp_test);
exec_time = toc;
fprintf('Execution time (joint GP) = %5.1f\n', exec_time);

full_gp.mnlp = mnlp_test;
full_gp.nmse = nmse_test;
full_gp.logtheta = logtheta;
full_gp.exec_time = exec_time;

%end
