using SCDM, LinearAlgebra

si_src_dir = "./test/scdm_dataset/Si"
STensor, UTensor, w_list, kplusb, Nk, Nb, Ne = load_problem(si_src_dir)

function make_f(STensor::Array{ComplexF64,4}, w_list, kplusb, Nk, Nb, Ne, Nj)
    R = zeros(ComplexF64, size(STensor))
    rho_hat = zeros(ComplexF64, Ne, Nb)
    omega = zeros(Float64, 1)
    function f(UTensor::Array{ComplexF64,3})
        f_oracle!(STensor, R, UTensor, w_list, kplusb, rho_hat, Nk, Nb, Ne, Nj, omega)
        return first(omega)
    end
end

function make_grad_f(STensor::Array{ComplexF64,4}, w_list, kplusb, Nk, Nb, Ne, Nj)
    R = zeros(ComplexF64, size(STensor))
    rho_hat = zeros(ComplexF64, Ne, Nb)
    omega = zeros(Float64, 1)
    grad_work = zeros(ComplexF64, Ne, Ne)
    grad_omega = zeros(ComplexF64, Ne, Ne, Nk)
    function grad_f(UTensor::Array{ComplexF64,3})
        f_oracle!(STensor, R, UTensor, w_list, kplusb, rho_hat, Nk, Nb, Ne, Nj, omega)
        grad_f_oracle!(R, w_list, rho_hat, Nk, Nb, Ne, Nj, grad_omega)
        SCDM.project!(UTensor, grad_omega, grad_work, Nk, Ne)
        return -grad_omega
    end
end

f = make_f(STensor, w_list, kplusb, Nk, Nb, Ne, Ne)
grad_f = make_grad_f(STensor, w_list, kplusb, Nk, Nb, Ne, Ne)

function quadratic_fit_1d(f::Function)
    a, negative_2ab, _ = [0.5 -1 0.5; -1.5 2 -0.5; 1.0 0.0 0.0] * f.(0:2)
    b = -negative_2ab / (2a)
    return b, f(b)
end

const ideg = 6

function retract(U, grad)
    U_buf = zeros(ComplexF64, size(U))
    copy!(U_buf, U)
    Ne, _, Nk = size(U)
    size_u_work = 4 * Ne * Ne + ideg + 1
    u_work = zeros(ComplexF64, size_u_work)
    SCDM.retract!(U_buf, grad, u_work, Nk, Ne, ideg, size_u_work)
    return U_buf
end


function make_step(U::Array{ComplexF64,3}, p_curr, lambda)
    new_U = deepcopy(U)
    for k in axes(p_curr, 3)
        new_U[:, :, k] = U[:, :, k] * cis(-Hermitian(1im * lambda * p_curr[:, :, k]))
    end
    return new_U
end
function line_search(U, f, grad_f, two_norm, step)
    f_curr, grad_curr = f(U), grad_f(U)
    res_curr = two_norm(grad_curr)

    while true
        Q = f_curr - 0.5step * res_curr
        V = make_step(U, grad_curr, step)
        f_v = f(V)
        f_v > Q || return V, f_v, step, res_curr
        step /= 2
    end
end

UTensor = zeros(ComplexF64, Ne, Ne, Nk)
for k in 1:Nk
    A = rand(Ne, Ne)
    Uu, S, V = svd(A)
    UTensor[:, :, k] = Uu * V'
end


function cg(UTensor::Array{ComplexF64,3}, f, grad_f, N::Int, Nk::Int)
    two_norm = x -> norm(x)^2
    iter = 0
    f_prev, grad_prev = f(UTensor), grad_f(UTensor)
    res_prev = two_norm(grad_prev)
    p_prev = zeros(ComplexF64, size(grad_prev))
    lambda_prev = 0.02

    while true
        res_prev / (Nk * N^2) < 1e-7 && break
        grad_curr = grad_f(UTensor)
        res_curr = two_norm(grad_curr)
        beta = rem(iter, N^2) == 0 ? 0 : res_curr / res_prev
        p_curr = grad_curr + beta * p_prev

        lambda_curr, f_curr = quadratic_fit_1d(lambda -> make_step(UTensor, p_curr, lambda) |> f)
        if lambda_curr > 0 && f_curr < f_prev
            UTensor = make_step(UTensor, p_curr, lambda_curr)
            f_curr = f(UTensor)
        else
            UTensor, f_curr, lambda_curr, _ = line_search(UTensor, f, grad_f, two_norm, 2lambda_prev)
        end

        lambda_prev = lambda_curr
        f_prev = f_curr
        p_prev = p_curr
        grad_prev = grad_curr
        res_prev = res_curr

        println(f(UTensor))
        iter += 1
    end
    println(iter)
    return UTensor
end


cg(UTensor, f, grad_f, Ne, Nk);
