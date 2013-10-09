classdef gp_grid_gpmodel_class < handle
    % The following properties can be set only by class methods
    properties (SetAccess = private)

    end
    properties (SetAccess = public)
        cov;
        noise_struct
        hyps_in_d
        hyperparams
        learn
        logpriorfuns
    end
    methods
       function gpmodel = gp_grid_gpmodel_class(varargin)
           
            if(~isempty(varargin) && isa(varargin{1},'gp_grid_gpmodel_class'))
                gpmodel_copyfrom = varargin{1};
                gpmodel.cov = gpmodel_copyfrom.cov;
                gpmodel.noise_struct =  gpmodel_copyfrom.noise_struct;
                gpmodel.hyps_in_d = gpmodel_copyfrom.hyps_in_d;
                gpmodel.hyperparams = gpmodel_copyfrom.hyperparams;
                gpmodel.learn = gpmodel_copyfrom.learn;
                gpmodel.logpriorfuns = gpmodel_copyfrom.logpriorfuns;
            end
       end
    end % methods
end % classdef
