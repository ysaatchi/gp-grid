function [Iout] = run_gpgrid_img_makespherical_noise(filename,datafolder,testfolder,savefolder,subdir,edges,mask)

loadmatfile = matfile([datafolder,filename,'.mat']);

frame0 = loadmatfile.data(:,:,1);
data = loadmatfile.data(edges(1):edges(2),edges(3):edges(4),:);
dir_name = filename;

img = mean(data,3);     %object
xsize = size(img,1);
ysize = size(img,2);

% figure; imagesc(img); return;

% take off elements by the boarders (2 pix)
h = ones(5,5)/25;
mask_filter = filter2(h,mask,'same');
mask = (mask_filter >= 0.999);

xsubindexstart = 1;
ysubindexstart = 1;


img_mask = img(logical(mask));
% noise_std = mean(img_mask(:))*0.05
noise_std = 15;
img = img+noise_std*gpml_randn(0.15, xsize, ysize);

xsubindex = xsubindexstart:2:xsize;
ysubindex = ysubindexstart:2:ysize;
mask_sub = mask(xsubindex,ysubindex);

sub_img = img(xsubindex,ysubindex);
sub_img_mask = sub_img(logical(mask_sub));
sub_img_mean = 0*mean(sub_img_mask(:));
sub_img = sub_img-sub_img_mean;

mv = mean(data,3);
vv = var(data,[],3);

p = [0   1];

esn = (p(1)*(sub_img+sub_img_mean)+p(2));

sn = zeros(size(sub_img));
imagesc(mask);

sn(logical(~mask_sub)) = 1500*max(sub_img(:));

sn = sn+esn;


Xgrid = cell(2,1);
Xgrid{2} = xsubindex;
Xgrid{1} = ysubindex;

startTime = rem(now,1);
initHypGuess.l = 2.^(1:3:7);
initHypGuess.sf = 2.^[1,5];
initHypGuess.sn = 0.05*mean(sub_img(:));
cov = {'covMaterniso', 5};
% cov = {'covMaterniso', 3};
% cov = {'covSEard'};

startTime = rem(now,1);
% [gpimg, logtheta_learned_dn,Stdfull] = gpgrid_img_dn(sub_img, Xgrid, xsize, ysize, sn(:), mask);
% [gpimg, logtheta_learned_dn] = gpgrid_img(sub_img,Xgrid, xsize, ysize);
[logtheta_learned, Iout, Stdfull] = gpgrid_img_dn(sub_img, Xgrid, [xsize, ysize], sn(:), mask, initHypGuess, cov);
exec_time = (rem(now,1)-startTime)*24*3600;

%%

run('makereport');

close all;
end
