using SCDM, LinearAlgebra, Random, CUDA

src_dir = "./test/scdm_dataset/BaTiO3"
s, u, w_list, k_plus_b, n_k, n_b, n_e = load_problem(src_dir)
u = CuArray(u)

f_gpu = make_f_gpu(s, w_list, k_plus_b, n_k, n_b, n_e, n_e)
grad_f_gpu = make_grad_f_gpu(f_gpu)

Random.seed!(16)
u = CUDA.zeros(ComplexF64, n_e, n_e, n_k)

for k in 1:n_k
    a = rand(n_e, n_e)
    u[:, :, k] = let (u, _, v) = svd(a)
        u * v'
    end
end

f_gpu(u)

@time cg(u, f_gpu, grad_f_gpu, retract!, n_e);
@time grad_f(u);
@profview cg(u, f, grad_f, n_e, n_k);
