%
% Elad Gilboa 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef gp_grid_input_class < handle
    % The following properties can be set only by class methods
    properties (SetAccess = private)
        xgrid;
        D;
    end
    properties (SetAccess = public)
        index_to_N;
        zeromeandata;
        meandata;
        Fs;
    end
    % Define an event called InsufficientFunds
    methods
        function gp_input = gp_grid_input_class(varargin)
            if(isa(varargin{1},'gp_grid_input_class'))
                gpinput_copyfrom = varargin{1};
                gp_input.xgrid=gpinput_copyfrom.xgrid;
                gp_input.D=gpinput_copyfrom.D;
                gp_input.index_to_N=gpinput_copyfrom.index_to_N;
                gp_input.zeromeandata=gpinput_copyfrom.zeromeandata;
                gp_input.meandata=gpinput_copyfrom.meandata;
                gp_input.Fs=gpinput_copyfrom.Fs;
            else
                Data = varargin{1};
                gp_input.index_to_N = varargin{2};
                gp_input.meandata = mean(Data(gp_input.index_to_N));
                gp_input.zeromeandata = Data(gp_input.index_to_N) - gp_input.meandata;
                if(length(varargin) < 3)
                    gp_input.Fs = 1;
                else
                    gp_input.Fs= varargin{3};
                end
                
                if(~isvector(Data))
                    Dim = length(size(Data));
                    Vecs = cell(1,Dim);
                    for d = 1:Dim
                        if(length(gp_input.Fs)==Dim)
                            Fs_d = gp_input.Fs(d);
                        else
                            Fs_d = gp_input.Fs;
                        end
                        Vecs{d} = Fs_d*(1:size(Data,d));
                    end
                    gp_input.make_xgrid(Vecs{:});
                end
            end
        end
        function copy_xgrid(gp_input, gpinput_copyfrom)
            if(isa(gpinput_copyfrom,'gp_grid_input_class'))
                gp_input.D = gpinput_copyfrom.D;
                gp_input.xgrid = gpinput_copyfrom.xgrid;
            end
        end
        function make_xgrid(gp_input, varargin)
            gp_input.D = length(varargin);
            gp_input.xgrid = fliplr(varargin(:)');
        end
        function data = get_data(gp_input)
            data = gp_input.zeromeandata(:)+ gp_input.meandata;
        end
        function N = get_N(gp_input)
            N = prod(cellfun(@length, gp_input.xgrid));
        end
        function n = get_n(gp_input)
            n = length(gp_input.index_to_N);
        end
        function D = get_D(gp_input)
            D = gp_input.D;
        end
        function oldmean = set_mean(gp_input, gpmean)
            oldmean = gp_input.meandata;
            gp_input.zeromeandata = gp_input.get_data()  - gpmean;
            gp_input.meandata = gpmean;
            
        end
        function [xstar] = make_xstar(gp_input,varargin)
            P = length(varargin);
            if(P == 1)
                ind_or_subs = varargin{1};
                if(isvector(ind_or_subs))
                    % xstar input is a vector of indices from xgrid
                    possible_xstar = makePossibleComb(gp_input.xgrid);
                    xstar = possible_xstar(ind_or_subs,:);
                else
                    % if xstar is in subs form then just need to flip the matrix in order
                    % to change it from Matlab matrix notation (most rapidly changing dimension
                    % is the first) to gp_grid notation (most rapidly changing dimension is
                    % the last)
                    xstar = fliplr(ind_or_subs);
                end
            else
                % need to build xstar from vectors of subs in each dimensions
                if(length(gp_input.xgrid) ~= P)
                    error('dimension misfit')
                end
                xstar = makePossibleComb(fliplr(varargin));
            end
        end
        function [tr_input, tst_input, cvxstar] = splitsets(gp_input, tstset, addednoise)
            
            if(isempty(tstset))
                tst_input = gp_grid_input_class(gp_input);
                tr_input = gp_grid_input_class(gp_input);
                tr_input.addgaussiannoise(addednoise);
                cvxstar = gp_input.make_xstar(gp_input.index_to_N);
            else 
                [cvxstar_indx, ~, cvindx_to_N] = intersect(tstset,gp_input.index_to_N);
                cvxstar = gp_input.make_xstar(cvxstar_indx);
                
                tst_input = gp_grid_input_class(gp_input.get_data(),cvxstar_indx,gp_input.Fs);
                tst_input.copy_xgrid(gp_input);
                
                tr_index_to_N = gp_input.index_to_N;
                tr_index_to_N(cvindx_to_N) = [];   %takeout indices that are in cv
                tr_input = gp_grid_input_class(gp_input.get_data(),tr_index_to_N,gp_input.Fs);
                tr_input.copy_xgrid(gp_input);
                tr_input.addgaussiannoise(addednoise);
            end
        end
        function res = isgridcomplete(gp_input)
            res = (gp_input.get_N()==gp_input.get_n());
        end
        function addgaussiannoise(gp_input,addednoise)
            newdata = gp_input.get_data()+addednoise*randn(size(gp_input.index_to_N));
            gp_input.meandata = mean(newdata(:));
            gp_input.zeromeandata = newdata(:)-gp_input.meandata;
        end
    end % methods
end % classdef
