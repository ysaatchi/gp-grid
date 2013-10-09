% Demo code to help understand how to use gp_grid for multidimension
% interpolation.
%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear variables;
clc;

cov = {'covSEard'};

% load img data
load('face_data');

xsize = size(img,1);
ysize = size(img,2);
xsubindex = 1:2:xsize;  % downsample indices of the image 
ysubindex = 1:2:ysize;


index_to_N = sub2ind_highD([xsize,ysize], makePossibleComb({xsubindex,ysubindex}));
Fs = 1; % sampling frequency. Unless given otherwise use 1.

% initilize an input class. This will automatically setup the grid
gp_input = gp_grid_input_class(img, index_to_N, Fs);


% use homoscedastic noise model. If want heteroscedastic noise model then
% use change noisevar to different noise variances however then you can not learn noise hyperparameter.
noisevar = ones(size(img));
noise_struct = gp_grid_noise_class(noisevar, gp_input.index_to_N);
noise_struct.learn = true;

% gpmodel class contain all the information of the model such as covarinace
% function, the noise structure, and hyperparameters.
gpmodel = gp_grid_gpmodel_class();
gpmodel.cov = cov;
gpmodel.noise_struct = noise_struct;

Qs = [2,2]; %how many hyperparameters in each dimension. For SE and Matern there are 2 in each dimension.
gpmodel.hyps_in_d  = make_hyps_in_d(Qs, cov);
gpmodel.learn=true;
maxiteration = 500; % for optimization

%use make_xstar for consistency with gp_grid notation
xstar = gp_input.make_xstar(1:numel(img));  

% setup initial guesses for hyperparameters. Here we use 2d covSEard kernel
% hence we have to initialize 4 hypers l1, l2, sf, sn. GPgrid uses the
% product of the kernels as the likelihood function. The signal std
% sf and noise std are joint hyperparameters for both dimensions hence will
% be multiplied in the likelihood function.
% for SE kernel, the product kernel will be
% K=[sf^2*exp(-(x-x')^2/2*l1)]*[sf^2*exp(-(x-x')^2/2*l2)]
gpmodel.hyperparams = log( ...
    [   1,      1,      sqrt(std(gp_input.zeromeandata)), -1      
        0.1,    0.1,    sqrt(std(gp_input.zeromeandata)), -1    ]);

[hypers_learned, trnlml, Iout] = gpgrid(gp_input, gpmodel, xstar, maxiteration);


% The grid axis order opposite to matlab.
tmpData = zeros(length(gp_input.xgrid{2}),length(gp_input.xgrid{1}));
tmpData(gp_input.index_to_N) = gp_input.get_data();
figure; imagesc(tmpData); colormap(gray)
title('before GP')

tmpData = zeros(length(gp_input.xgrid{2}),length(gp_input.xgrid{1}));
tmpData(1:numel(tmpData)) = Iout+gp_input.meandata;
figure; imagesc(tmpData); colormap(gray)
title('After GP')
