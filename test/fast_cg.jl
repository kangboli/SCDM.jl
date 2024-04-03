using SCDM, LinearAlgebra, Random, Profile

si_src_dir = "./test/scdm_dataset/Si"
STensor, UTensor, w_list, kplusb, Nk, Nb, Ne = load_problem(si_src_dir)

struct OracleF
    STensor::Array{ComplexF64, 4}
    w_list::Vector{Float64}
    kplusb::Matrix{Int64}
    Nk::Int64
    Nb::Int64
    Ne::Int64
    Nj::Int64
    R::Array{ComplexF64,4}
    rho_hat::Matrix{ComplexF64}
    omega::Vector{Float64}
end

function make_f(STensor::Array{ComplexF64,4}, w_list, kplusb, Nk, Nb, Ne, Nj)
    R = zeros(ComplexF64, size(STensor))
    rho_hat = zeros(ComplexF64, Ne, Nb)
    omega = zeros(Float64, 1)
    OracleF(STensor, w_list, kplusb, Nk, Nb, Ne, Nj, R, rho_hat, omega)
    # function f(UTensor::Array{ComplexF64,3})
    #     f_oracle!(STensor, R, UTensor, w_list, kplusb, rho_hat, Nk, Nb, Ne, Nj, omega)
    #     return first(omega)
    # end
end

function (f::OracleF)(UTensor::Array{ComplexF64,3})
    f_oracle!(f.STensor, f.R, UTensor, f.w_list, f.kplusb, f.rho_hat, f.Nk, f.Nb, f.Ne, f.Nj, f.omega)
    return first(f.omega)
end

struct OracleGradF
    f::OracleF
    grad_work::Matrix{ComplexF64}
    grad_omega::Array{ComplexF64,3}
end

function make_grad_f(f::OracleF)
    # R = zeros(ComplexF64, size(STensor))
    # rho_hat = zeros(ComplexF64, Ne, Nb)
    # omega = zeros(Float64, 1)
    grad_work = zeros(ComplexF64, Ne, Ne)
    grad_omega = zeros(ComplexF64, Ne, Ne, Nk)
    return OracleGradF(f, grad_work, grad_omega)
    # function grad_f!(UTensor::Array{ComplexF64,3})
    #     f_oracle!(STensor, R, UTensor, w_list, kplusb, rho_hat, Nk, Nb, Ne, Nj, omega)
    #     grad_f_oracle!(R, w_list, rho_hat, Nk, Nb, Ne, Nj, grad_omega)
    #     SCDM.project!(UTensor, grad_omega, grad_work, Nk, Ne)
    #     return grad_omega
    # end
end

function (grad_f::OracleGradF)(UTensor::Array{ComplexF64,3})
    f = grad_f.f
    grad_f_oracle!(f.R, f.w_list, f.rho_hat, f.Nk, f.Nb, f.Ne, f.Nj, grad_f.grad_omega)
    SCDM.project!(UTensor, grad_f.grad_omega, grad_f.grad_work, f.Nk, f.Ne)
    return grad_f.grad_omega

end

f = make_f(STensor, w_list, kplusb, Nk, Nb, Ne, Ne)
grad_f = make_grad_f(f)

function quadratic_fit_1d(f::Function)
    a, negative_2ab, _ = [0.5 -1 0.5; -1.5 2 -0.5; 1.0 0.0 0.0] * f.(0:2)
    b = -negative_2ab / (2a)
    return b, f(b)
end

ideg = 4
#=
function retract(u_buffer, u_tensor, p_curr, t::Float64)
    t == 0 && return u_tensor
    copy!(u_buffer, u_tensor)
    Ne, _, Nk = size(u_tensor)
    size_u_work = 4 * Ne * Ne + ideg + 1
    u_work = zeros(ComplexF64, size_u_work)
    SCDM.retract!(u_buffer, p_curr, u_work, Nk, Ne, ideg, size_u_work, t)
    return u_buffer
end =#


function make_step!(u_buffer, u_tensor::Array{ComplexF64,3}, p_curr, lambda)
    lambda == 0 && return u_tensor
    copy!(u_buffer, u_tensor)
    Ne, _, Nk = size(u_tensor)
    size_u_work = 4 * Ne * Ne + ideg + 1
    u_work = zeros(ComplexF64, size_u_work)
    SCDM.retract!(u_buffer, p_curr, u_work, Nk, Ne, ideg, size_u_work, float(-lambda))
    return u_buffer
end

function line_search!(u_buffer, u_tensor, f, f_curr, grad_curr, res_curr, step)
    #= u_buffer = zeros(ComplexF64, size(u_tensor)) =#
    while true
        Q = f_curr - 0.5step * res_curr
        make_step!(u_buffer, u_tensor, grad_curr, step)
        f_v = f(u_buffer)
        f_v > Q || return f_v, step, res_curr
        step /= 2
    end
end

#= lambda_curr, f_tmp = quadratic_fit_1d(lambda -> make_step!(u_buffer, u_tensor, p_curr, lambda) |> f) =#
# scale, f_tmp = quadratic_fit_1d(scale ->
#     scale == 0 ? f_curr : f(make_step!(u_buffer, u_tensor, p_curr, scale * lambda_prev)))

"""

Memory consumption:

The current implementation keeps two copies of the overlap matrices to speed up
the gradient calculation by reusing the intermediate values from the forward
evaluation. Whether this tradeoff is worthwhile depends on how often the gradient
is evaluated compared to the objective function, which depends on the optimizer.
Exploring the tradeoff is currently not a priority.
"""

function cg(u_tensor::Array{ComplexF64,3}, f_no_wrap, grad_f!, N::Int, Nk::Int)
    normsq = x -> norm(x)^2
    total_eval = 0
    function f(u)
        total_eval += 1
        return f_no_wrap(u)
    end
    f_prev, grad_prev = f(u_tensor), grad_f!(u_tensor)
    res_prev = normsq(grad_prev)
    p_prev = zeros(ComplexF64, size(u_tensor))
    u_buffer = zeros(ComplexF64, size(u_tensor))
    iter = 0
    lambda_prev = 1
    f_curr = f_prev

    while res_prev / (Nk * N^2) > 1e-7
        grad_curr = grad_f!(u_tensor)
        res_curr = normsq(grad_curr)
        # The fletcher reeves 
        beta = rem(iter, N^2) == 0 ? 0 : res_curr / res_prev
        p_curr = grad_curr
        LinearAlgebra.axpy!(beta, p_prev, p_curr)

        function try_scale(scale)
            scale == 0 && return f_curr
            return f(make_step!(u_buffer, u_tensor, p_curr, scale * lambda_prev))
        end
        scale, f_tmp = quadratic_fit_1d(try_scale)
        lambda_curr = scale * lambda_prev
        if lambda_curr > 0 && f_tmp < f_prev
            copy!(u_tensor, u_buffer)
            f_curr = f_tmp
        else # fall back to a gradient descent if the function is not locally convex.
            LinearAlgebra.axpy!(-beta, p_prev, p_curr)
            f_curr, lambda_curr, _ = line_search!(u_buffer, u_tensor, f, f_curr, p_curr, res_curr, 2lambda_prev)
            LinearAlgebra.axpy!(beta, p_prev, p_curr)
            copy!(u_tensor, u_buffer)
        end

        lambda_prev = lambda_curr
        f_prev = f_curr
        copy!(p_prev, p_curr)
        res_prev = res_curr

        println(f_curr)
        iter += 1
    end
    println(iter)
    println(total_eval)
    return u_tensor
end

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