
using SCDM, LinearAlgebra, Random, CUDA, WTP

src_dir = "./test/scdm_dataset/N2Na2O6"
wave_functions_list = wave_functions_from_directory(joinpath(src_dir, "aiida.save"))
k_map, brillouin_zone = i_kpoint_map(wave_functions_list)
mmn = MMN(joinpath(src_dir, "aiida.mmn"))
ki = NeighborIntegral(mmn, k_map)
s, u, w_list, k_plus_b, k_minus_b, n_k, n_b, n_e, shell_list = load_problem(src_dir)

for k in brillouin_zone
    for (i, b) in enumerate(shell_list)
        s[:, :, linear_index(k), i] = ki[k, k + b]
    end
end

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

