function [gpimg] = run_gpgrid_img_spherical(filename,datafolder,testfolder,edges,ineq,thresh,closeholes)


% %%%%%%%% HORSE
% I0_big=load('aug304_0');
% dir_name = 'horse';
% frame0=I0_big.aug25_0(480:580,780:880);
% data = frame0;
% [mask] = (frame0>800);%makemask(frame0.*(frame0<800), 0.6,2000,5);

% ============================= PCB ===========================\
% filename = 'sphere_0';
% load(['C:\Users\Elad\Documents\Research\phaseCam\big data\',filename,'.mat'])
% frame0 = data(:,:,1);
% dir_name = filename;
% %
% % img = frame0(90:190,50:150);     %object
%
% img = frame0(1:200,1:200);
% data = data(1:200,1:200,:);
% mask=zeros(size(img));
% mask = img<300;

% ============================= Bottle ===========================\

loadmatfile = matfile([datafolder,filename,'.mat']);


frame0 = loadmatfile.data(:,:,1);
data = loadmatfile.data(edges(1):edges(2),edges(3):edges(4),:);
dir_name = filename;

img = frame0(edges(1):edges(2),edges(3):edges(4),1);     %object
% data = data(181:340,241:400,1:900);

mask=zeros(size(img));
% mask = img > 1500;          %text
if(ineq < 0)
    mask = img < thresh;            %background
elseif(ineq > 0)
    mask = img > thresh;            %object
end


if(closeholes)
    mask = bwareaopen(mask,1000);
    mask = 1-bwareaopen(1-mask,1000);
end
xsize = size(img,1);
ysize = size(img,2);

sub_img = img(1:2:xsize,1:2:ysize);
mv = mean(data,3);
vv = var(data,[],3);
p=polyfit(mv(:),vv(:),1);
% % maskIndx = find(mask == 0);
% % p=polyfit(mv(maskIndx),vv(maskIndx),1);
esn = (p(1)*sub_img+p(2));
if(min(esn(:))<=0)
    y = (p(1)*max(sub_img(:))+p(2));
    p(1) = (y-min(vv(:)))/(max(mv(:))-min(mv(:)));
    p(2) = min(vv(:));
    % %     y = (p(1)*max(sub_img(maskIndx))+p(2));
    % %     p(1) = (y-min(vv(maskIndx)))/(max(mv(maskIndx))-min(mv(maskIndx)));
    % %     p(2) = min(vv(maskIndx));
    % %     esn = (p(1)*sub_img+p(2));
    %     esn = (y-min(vv(:)))/(max(mv(:))-min(mv(:)))*sub_img+min(vv(:));
    %     (p(1)*sub_img+p(2))-min(esn(:))+min(vv(:));       %correction to noise model so will not get negative vlues
end

sn = zeros(size(sub_img));
imagesc(mask);
mask_sub = mask(1:2:xsize,1:2:ysize);
sn(logical(mask_sub)) = 3000*max(sub_img(:));%0*max(max(B.*(1-mask_sub)));
sn = sn+esn;



Xgrid = cell(2,1);
Xgrid{2} = 1:2:xsize;
Xgrid{1} = 1:2:ysize;

startTime = rem(now,1);
% [gpimg, logtheta_learned_dn,Stdfull] = gpgrid_img_dn(sub_img, Xgrid, xsize, ysize, sn(:), mask);
[gpimg, logtheta_learned_dn] = gpgrid_img(sub_img,Xgrid, xsize, ysize);
exec_time = (rem(now,1)-startTime)*24*3600;

%%
% close all;

start_time = datestr(now,30);
while(isempty(dir_name))
    dir_name = input('Directory name: ', 's');
end

% while(1)
%     svfolder = ['C:/Users/Elad/Documents/Research/phaseCam/tests/',dir_name,'/'];
%     mkdir(svfolder)
%     if (isequal(lastwarn,'Directory already exists.'))
%         reply = input('Directory already exists. Do you want to continue? Y/N [Y]: ', 's');
%         if ~isequal(reply,'Y')
%             reply = 'Y';
%             break;
%         end
%     else
%         break;
%     end
% end

%%
fntsz = 30;
svfolder = [testfolder,dir_name,'/',start_time,'-spherical/'];
mkdir(svfolder)

%%
copyfile([testfolder,'showFigureseEmpty.tex'],[svfolder,'showFigures.tex'])
fid = fopen([svfolder,'showFigures.tex'], 'a');


fprintf(fid, '\\section{%s}\n',strrep(dir_name, '_', ' '));
fprintf(fid, 'Spherical noise!! \\\\ \n');
fprintf(fid, 'exec time: %5.2f, \n',exec_time);
fprintf(fid, 'theta= (%3.2f, %3.2f, %3.2f, %3.2f)\n',exp(logtheta_learned_dn(1)),exp(logtheta_learned_dn(2)),exp(logtheta_learned_dn(3)),exp(logtheta_learned_dn(4)));

imageTrue = mean(data,3);
new_mask = mask;
new_mask(1:5,:)=1;
new_mask(:,1:5)=1;
new_mask(end-4:end,:)=1;
new_mask(:,end-4:end)=1;
imageTrueN = imageTrue.*(1-new_mask);

num_non_zero = sum(1-new_mask(:));
% new_mask = (imageTrue<160);
% new_mask = (imageTrue>2500);
figIdx = 1;


figure; imshow(mat2gray(imageTrue)); title('True image','FontSize',fntsz);
figureName = 'trueImage';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.25)

figure; imshow(mat2gray(1-new_mask)); title('mask','FontSize',fntsz);
figureName = 'mask';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.25)

figure; imshow(mat2gray(imageTrue.*(1-new_mask))); title('True image (after mask)','FontSize',fntsz);
figureName = 'trueImageAfterMaskGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure;  imagesc((imageTrue.*(1-new_mask))); title('True image after mask)','FontSize',fntsz);
figureName = 'trueImageAfterMask';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)

figure;plot(mv(:),vv(:),'.'); title('variation vs mean','FontSize',fntsz);
hold on;
plot(sub_img(:),esn(:),'r.')
figureName = 'varVsMean';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)


% figureName = 'invKy';figIdx = figIdx+1;
% if(sum(findobj('Type','figure') == 11))     %is figure 11 open?
%     figure(11); colorbar;
%     print( gcf , '-depsc' , [svfolder,figureName ]);
% end
% writeFigure(fid,[figureName,'.eps'],0.3)

err_gp =  sum(sum(((imageTrue-gpimg).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
psnr_gpimg =  10*log10(255^2/sum(sum( ((( double( uint8(255*mat2gray(imageTrueN.*(1-new_mask))) - uint8(255*mat2gray(gpimg.*(1-new_mask))) ) ).^2).*(1-new_mask)/num_non_zero))) )

figure; imshow(mat2gray(gpimg.*(1-new_mask)));title('GP interpolation','FontSize',fntsz); xlabel(['SMSE GP = ',num2str(err_gp)],'FontSize',fntsz);
figureName = 'GPimgGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((gpimg.*(1-new_mask)));title('GP interpolation','FontSize',fntsz); xlabel(['SMSE GP = ',num2str(err_gp)],'FontSize',fntsz);
figureName = 'GPimg';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)

I0_sub = sub_img(1:1:end,1:1:end);
I0_m2 = abs(cbsi (I0_sub,1));
I0_m2 = I0_m2(1:size(imageTrue,1),1:size(imageTrue,2));
err_bicsp =  sum(sum(((imageTrue-I0_m2).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
figure; imshow(mat2gray(I0_m2.*(1-new_mask)));title('Bicubic-Spline interpolation','FontSize',fntsz); xlabel(['SMSE BICSP = ',num2str(err_bicsp)],'FontSize',fntsz);
figureName = 'BICSPimgGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((I0_m2.*(1-new_mask)));title('Bicubic-Spline interpolation','FontSize',fntsz); xlabel(['SMSE BICSP = ',num2str(err_bicsp)],'FontSize',fntsz);
figureName = 'BICSPimg';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)

I0_sub = sub_img(1:1:end,1:1:end);
I0_m2 = abs(cbsi (I0_sub,1));
I0_m2 = I0_m2(1:size(imageTrue,1),1:size(imageTrue,2));
h = fspecial('average', 3);
I0_m2 = filter2(h, I0_m2);
err_bicsp =  sum(sum(((imageTrue-I0_m2).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
figure; imshow(mat2gray(I0_m2.*(1-new_mask)));title('Bicubic-Spline interpolation (low pass after interpolation)','FontSize',fntsz); xlabel(['SMSE BICSP = ',num2str(err_bicsp)],'FontSize',fntsz);
figureName = 'BICSPimgLPassAfterGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((I0_m2.*(1-new_mask)));title('Bicubic-Spline interpolation (low pass after interpolation)','FontSize',fntsz); xlabel(['SMSE BICSP = ',num2str(err_bicsp)],'FontSize',fntsz);
figureName = 'BICSPimgLPassAfter';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)

h = fspecial('average', 3);
BSmoothed = filter2(h, sub_img);
I0_sub = BSmoothed(1:1:end,1:1:end);
I0_m2 = abs(cbsi (I0_sub,1));
I0_m2 = I0_m2(1:size(imageTrue,1),1:size(imageTrue,2));
err_bicsp =  sum(sum(((imageTrue-I0_m2).^2)./(p(1)*imageTrue+p(2)).*(1-new_mask)))/num_non_zero
figure; imshow(mat2gray(I0_m2.*(1-new_mask)));title('smooth Bicubic-Spline interpolation (low pass before interpolation)','FontSize',fntsz); xlabel(['SMSE BICSP = ',num2str(err_bicsp)],'FontSize',fntsz);
figureName = 'BICSPimgLPassBeforerGr';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
% writeFigure(fid,[figureName,'.eps'],0.3)

figure; imagesc((I0_m2.*(1-new_mask)));title('smooth Bicubic-Spline interpolation (low pass before interpolation)','FontSize',fntsz); xlabel(['SMSE BICSP = ',num2str(err_bicsp)],'FontSize',fntsz);
figureName = 'BICSPimgLPassBeforer';figIdx = figIdx+1;
print( gcf , '-depsc' , [svfolder,figureName ]);
writeFigure(fid,[figureName,'.eps'],0.3)

save([svfolder,'workspace'])

fprintf(fid, '\\end{document}\n');

fclose(fid);

oldFolder = cd(svfolder);
system('copy "showFigures.tex" "showFigures1.tex"');
system('latex.exe --src "showFigures1.tex"');
system('dvipdfmx.exe "showFigures1.dvi"');
% pause(1)
% system('"C:\Program Files (x86)\Adobe\Reader 10.0\Reader\AcroRd32.exe" showFigures1.pdf');
cd(oldFolder)

close all;
end
