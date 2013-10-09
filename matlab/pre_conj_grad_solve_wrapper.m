function [alpha_kron rs] = pre_conj_grad_solve_wrapper(Qs, V_kron, noise, Kstar_kron, C, max_iter,index_to_N);
Kstar_kron_input.data = Kstar_kron(index_to_N);
Kstar_kron_input.index_to_N = index_to_N;
[alpha_kron rs] = pre_conj_grad_solve(Qs, V_kron, noise, Kstar_kron_input, C, max_iter);
end