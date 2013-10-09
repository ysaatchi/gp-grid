function [Iout] = run_gpml_img_dn(filename,datafolder,testfolder,savefolder,subdir,edges,mask,params)

loadmatfile = matfile([datafolder,filename,'.mat']);

frame0 = loadmatfile.data(edges(1):edges(2),edges(3):edges(4),1);
data = loadmatfile.data(edges(1):edges(2),edges(3):edges(4),2:end);
dir_name = filename;
xsize = size(data,1);
ysize = size(data,2);

D = 2;

if(exist('params','var') && isfield(params,'makespherical') && params.makespherical == true)
    img = mean(data,3);
    if(isfield(params,'noise_std') && params.noise_std > 0)
        noise_std = params.noise_std;
    else
        noise_std = 0.01*mean(img(:));
    end    
    img = img+noise_std*gpml_randn(0.15, xsize, ysize);
    pactual= [0   1];
else
    % don't use noise_std in report
    noise_std = 1;
    img = frame0; 
    pactual = [0.2495   15.9858];
end

if(exist('params','var') && isfield(params,'sphericalnoise') && params.sphericalnoise == true)
    % spherical noise model
    pmodel = [0   1];
else
    % camera specific diagonal noise model
    pmodel = [0.2495   15.9858];
end

if(~exist('params','var') || ~isfield(params,'useboarders') || params.useboarders == false)
    % take off elements by the boarders (2 pix)
    h = ones(5,5)/25;
    mask_filter = filter2(h,mask,'same');
    mask = (mask_filter >= 0.999);
end

xsubindexstart = 1;
ysubindexstart = 1;

xsubindex = xsubindexstart:2:xsize;
ysubindex = ysubindexstart:2:ysize;
mask_sub = mask(xsubindex,ysubindex);

sub_img = img(xsubindex,ysubindex);
sub_img_mask = sub_img(logical(mask_sub));
% calculate the mean of the masked segment only
sub_img_mean = mean(sub_img_mask(:));
% remove the mean
sub_img = sub_img-sub_img_mean;


input.index_to_N = find(mask_sub == 1);
input.data = sub_img(input.index_to_N);

mv = mean(data,3);
vv = var(data,[],3);
pfit=polyfit(mv(:),vv(:),1);

esn = (pmodel(1)*(sub_img+sub_img_mean)+pmodel(2));
if(min(esn(:))<=0)
    pmodel(2) = pmodel(2)+min(esn); 
    esn = (pmodel(1)*(sub_img+sub_img_mean)+pmodel(2));
end

sn = zeros(size(sub_img));
imagesc(mask);

sn(logical(~mask_sub)) = 300*max(sub_img_mask(:));%0*max(max(B.*(1-mask_sub)));
sn = sn+esn;


Xgrid = cell(2,1);
Xgrid{2} = xsubindex;
Xgrid{1} = ysubindex;

if(exist('params','var') && isfield(params,'initHypGuess'))
    initHypGuess.l = params.initHypGuess.l;
    initHypGuess.sf = params.initHypGuess.sf;
    if(strcmp(params.initHypGuess.sn,'avg'))
        initHypGuess.sn = [0.01,0.05]*mean(sub_img(:)+sub_img_mean);
    else
        initHypGuess.sn = params.initHypGuess.sn;
    end
else
    initHypGuess.l = exp(2:2:7);
    initHypGuess.sf = exp([2,5]);
end

if(exist('params','var') && isfield(params,'cov'))
    cov = params.cov;
else
    cov = {'covMaterniso', 1};
    % cov = {'covMaterniso', 3};
    % cov = {'covSEard'};
end

startTime = rem(now,1);

% [logtheta_learned, Iout, Vout] = gpgrid_img_dn(sub_img, Xgrid, [xsize, ysize], sn(:), mask, initHypGuess, cov);
[x_starvec,y_starvec] = meshgrid(1:xsize,1:ysize);
xstar = [y_starvec(:),x_starvec(:)];
[x_vec,y_vec] = meshgrid(xsubindex,ysubindex);
x = [y_vec(:),x_vec(:)];

%%
% % [logtheta_learned, Iout, Vout] = gpgrid_dn(sensorData_work, Xgrid, [xsize, ysize, zsize], sn(:), xstar, initHypGuess, cov);
% [logtheta_learned, Iout] = gpgrid_dn(input, Xgrid, [xsize, ysize], sn(:), xstar, initHypGuess, cov);
% [logtheta_learned, Iout] = gpml_image(input, x, [xsize, ysize], sn(:), xstar, initHypGuess, cov);


% add back the mean
Iout_gp = Iout+sub_img_mean;

exec_time = (rem(now,1)-startTime)*24*3600;
p = pactual;
sub_img = img(xsubindex,ysubindex);

run('makereport')

close all;
end
