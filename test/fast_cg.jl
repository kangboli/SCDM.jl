using SCDM, LinearAlgebra, Random, Profile

si_src_dir = "./test/scdm_dataset/Si"
STensor, UTensor, w_list, kplusb, Nk, Nb, Ne = load_problem(si_src_dir)

f = make_f(STensor, w_list, kplusb, Nk, Nb, Ne, Ne)
grad_f = make_grad_f(f)

Random.seed!(16)
UTensor = zeros(ComplexF64, Ne, Ne, Nk)
for k in 1:Nk
    A = rand(Ne, Ne)
    Uu, S, V = svd(A)
    UTensor[:, :, k] = Uu * V'
end

@time cg(UTensor, f, grad_f, Ne, Nk);
@time f(UTensor)
@time grad_f(UTensor);
@profview cg(UTensor, f, grad_f, Ne, Nk);

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
