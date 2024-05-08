using SCDM, LinearAlgebra, Random
using CUDA

si_src_dir = "./test/scdm_dataset/Si"
s, u, w_list, k_plus_b, n_k, n_b, n_e = load_problem(si_src_dir)

f_gpu = make_f_gpu(s, w_list, k_plus_b, n_k, n_b, n_e, n_e)
grad_f_gpu = make_grad_f_gpu(f_gpu)

Random.seed!(16)
u_gpu = CUDA.zeros(ComplexF64, n_e, n_e, n_k)
for k in 1:n_k
    A = CUDA.rand(n_e, n_e)
    u_gpu[:, :, k] = let (u, _, v) = svd(A)
        u * v'
    end
end

CUDA.@profile cg(u_gpu, f_gpu, grad_f_gpu, retract_gpu!, n_e);
CUDA.@profile f_gpu(u_gpu)
f_gpu(u_gpu)
@time grad_f_gpu(u_gpu);
@profview cg(u, f, grad_f, n_e, n_k);
