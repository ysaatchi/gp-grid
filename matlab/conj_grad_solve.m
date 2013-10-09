function [x, residuals] = conj_grad_solve(Qs, V, noise, b)

%Qs : per-dimension eigenvector matrix (kronecker component)
%V_kron : eigenvalue vector (kronecker component)
%noise : noise vector (corresponds to diagonal noise)

epsilon = 1e-4; %used to assess convergence of conjugate gradient solver 

D = length(Qs);
QTs = cell(D,1);
for d = 1:D
    QTs{d} = Qs{d}';
end

%Taken almost verbatim from Golub and van Loan pp. 527
numIter = 3000;
N = length(b);
k = 0;
r = b; %initial residual -- used to assess convergence
x = zeros(N,1);
rho_b = sqrt(sum(b.^2));
rho_r = Inf;
residuals = zeros(numIter,1);
while ((rho_r > (epsilon*rho_b)) && (k < numIter))
	%move index
	k = k+1;
	if k > 1
		p_prev = p;
		r_prev_prev = r_prev;
	end
	r_prev = r;
	x_prev = x;
	
	%iteration core
	if k == 1
		p = r_prev;
	else
		beta = (r_prev'*r_prev)/(r_prev_prev'*r_prev_prev);
		p = r_prev + beta*p_prev;
	end
	%Compute (Q'*V*Q + diag(noise))*p
    Ap = kron_mv(QTs, p);
    Ap = Ap.*V;
    Ap = kron_mv(Qs, Ap);
    Ap = Ap + noise.*p;
    
	alpha = (r_prev'*r_prev)/(p'*Ap);
	x = x_prev + alpha*p;
	r = r_prev - alpha*Ap;
    rho_r = sqrt(sum(r.^2));
% 	fprintf('Residual = %3.3f\n', rho_r);
    residuals(k) = rho_r;
    
%     if k == 3000
%         keyboard;
%     end
end

fprintf('Conjugate gradient solver converged in %i iterations.\n',k); 
	
	
	 
	
