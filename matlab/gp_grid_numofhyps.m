function nhyps = gp_grid_numofhyps(hyps_in_dim)
    
nhyps = length(unique(cell2mat(hyps_in_dim)));

