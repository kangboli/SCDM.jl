using SCDM, LinearAlgebra, Random, CUDA

src_dir = "./test/scdm_dataset/BaTiO3/aiida.save"
s, u_gpu, w_list, k_plus_b, k_minus_b, n_k, n_b, n_e, _ = load_problem(src_dir, collect(12:20))
u_gpu = CuArray(Array{ComplexF32, 3}(u_gpu))

f_gpu = make_f_gpu(s, w_list, k_plus_b, k_minus_b, n_k, n_b, n_e, n_e)
grad_f_gpu = make_grad_f_gpu(f_gpu)

Random.seed!(16)
u_gpu = CUDA.zeros(ComplexF32, n_e, n_e, n_k)

for k in 1:n_k
    a = rand(n_e, n_e)
    u_gpu[:, :, k] = let (u, _, v) = svd(a)
        u * v'
    end
end


@time cg(u_gpu, f_gpu, grad_f_gpu, retract_gpu!, n_e);

