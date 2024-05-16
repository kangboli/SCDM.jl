using SCDM, LinearAlgebra, Random, Profile

src_dir = "./test/scdm_dataset/Si"
s, u, w_list, k_plus_b, k_minus_b, n_k, n_b, n_e, shell_list = load_problem(src_dir)

f = make_f(s, w_list, k_plus_b, n_k, n_b, n_e, n_e)
grad_f = make_grad_f(f)

Random.seed!(16)
u = zeros(ComplexF64, n_e, n_e, n_k)

for k in 1:n_k
    a = rand(n_e, n_e)
    u[:, :, k] = let (u, _, v) = svd(a)
        u * v'
    end
end

@time cg(u, f, grad_f, retract!, n_e);
