clear;
Ds = 8:20;
exec_times = zeros(size(Ds));
exec_times_full = zeros(size(Ds));

i = 1;

for D = Ds
    
    fprintf('D = %i\n', D);
    
    logtheta = log([ones(D+1,1);0.1]);
    jitter = 1e-10;
    for d = 1:D
        xg{d} = [-1, 1];
        K_kron = feval('covSEardOLD', [logtheta(d); logtheta(D+1)/D], xg{d}'); 
        L = chol(K_kron, 'lower');
        Lc{d} = L;
    end
    N = 2^D;
    rr = randn(N,1);
    sample_kron = kron_mv(Lc, rr);
    noise = rand(N,1)*0.1;
    y = sample_kron + exp(2*logtheta(end));
    
    tic;
    [nlml, dnlml] = gpr_covSEard_grid(logtheta, xg, y); 
    exec_times(i) = toc;
    
    if N < 8000
        tic;
        [nlml, dnlml] = gpr_covSEard(logtheta, xg, y);
        exec_times_full(i) = toc;
    end
    
    i = i+1;
       
end

hf = figure; 
fntsz=14;
loglog(2.^(Ds), exec_times, 'k', 'MarkerSize', 5);
idx = find(exec_times_full > 0);
hold on; loglog(2.^(Ds(idx)), exec_times_full(idx), 'r^', 'MarkerSize', 5);
xlabel('N = 2^{D}','fontsize',fntsz);
ylabel('Runtime (s)','fontsize',fntsz);
set(gca,'fontsize',fntsz);

getPDF(hf, 'kron-runtimes');
!mv kron-runtimes.pdf ~/PhD/doc/thesis/Chapter4/figures/
