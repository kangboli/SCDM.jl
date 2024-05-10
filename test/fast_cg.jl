using SCDM, LinearAlgebra, Random, Profile

src_dir = "./test/scdm_dataset/BaTiO3"
s, u, w_list, k_plus_b, k_minus_b, n_k, n_b, n_e = load_problem(src_dir)

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
@time f(u)
df = grad_f(u)
u_buffer = similar(u);
retract!(u_buffer, u, df, -0.01, QRRetraction());
f(u_buffer)
@time grad_f(u);
@profview cg(u, f, grad_f, n_e, n_k);

#= new_U = deepcopy(U)
for k in axes(p_curr, 3)
    new_U[:, :, k] = U[:, :, k] * cis(-Hermitian(1im * lambda * p_curr[:, :, k]))
end
return new_U =#
#=
function make_step(U::Array{ComplexF64,3}, p_curr, lambda)
    #= new_U = deepcopy(U)
    for k in axes(p_curr, 3)
        new_U[:, :, k] = U[:, :, k] * cis(-Hermitian(1im * lambda * p_curr[:, :, k]))
    end
    return new_U =#
    U_buf = zeros(ComplexF64, size(U))
    return retract(U_buf, U, p_curr, float(-lambda))
end =#

# lambda_curr, f_tmp = quadratic_fit_1d(lambda -> make_step!(u_buffer, u_tensor, p_curr, lambda) |> f)
