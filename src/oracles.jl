export OracleF, OracleGradF, make_f, make_grad_f, retract!, QRRetraction, SVDRetraction

struct OracleF
    s::Array{ComplexF64,4}
    w_list::Vector{Float64}
    k_plus_b::Matrix{Int64}
    n_k::Int64
    n_b::Int64
    n_e::Int64
    n_j::Int64
    r::Array{ComplexF64,4}
    rho_hat::Matrix{ComplexF64}
    omega::Vector{Float64}
    m_work::Array{ComplexF64,3}
end

function make_f(s::Array{ComplexF64,4}, w_list, kplusb, n_k, n_b, n_e, n_j)
    r = zeros(ComplexF64, size(s))
    rho_hat = zeros(ComplexF64, n_e, n_b)
    omega = zeros(Float64, n_b)
    m_work = zeros(ComplexF64, n_e, n_k, n_b)
    OracleF(s, w_list, kplusb, n_k, n_b, n_e, n_j, r, rho_hat, omega, m_work)
end

function (f::OracleF)(u::Array{ComplexF64,3})
    fill!(f.rho_hat, 0)
    fill!(f.m_work, 0)
    Threads.@threads for k in 1:f.n_k
        for b in 1:f.n_b
            @inbounds mul!(view(f.r, :, :, k, b), view(f.s, :, :, k, b), view(u, :, :, f.k_plus_b[k, b]))
        end
        for n in 1:f.n_e
            for b in 1:f.n_b
                @inbounds f.m_work[n, k, b] = dot(view(u, :, n, k), view(f.r, :, n, k, b))
            end
        end
    end

    Threads.@threads for n in 1:f.n_e
        for k in 1:f.n_k
            for b in 1:f.n_b
                @inbounds f.rho_hat[n, b] += f.m_work[n, k, b]
            end
        end
    end
    lmul!(1 / f.n_k, f.rho_hat)

    for b in 1:f.n_b
        #= f.omega[b] = 2 * f.w_list[b] * (f.n_e - sum(real, view(f.m_work, :, b))) =#
        f.omega[b] = 2 * f.w_list[b] * (f.n_e - sum(abs.(view(f.rho_hat, :, b))))
    end

    return sum(f.omega)
end

struct OracleGradF
    f::OracleF
    grad_work::Matrix{ComplexF64}
    grad_omega::Array{ComplexF64,3}
end

function make_grad_f(f::OracleF)
    grad_work = zeros(ComplexF64, f.n_e, f.n_e)
    grad_omega = zeros(ComplexF64, f.n_e, f.n_e, f.n_k)
    return OracleGradF(f, grad_work, grad_omega)
end

function anti_symmetrize!(dst, src)
    for i in axes(src, 1)
        for j in axes(src, 2)
            dst[i, j] = (src[i, j] - src[j, i]') / 2
        end
    end
    return dst
end

function (grad_f::OracleGradF)(u::Array{ComplexF64,3})
    f = grad_f.f
    #= grad_f_oracle!(f.r, f.w_list, f.rho_hat, f.n_k, f.n_b, f.n_e, f.Nj, grad_f.grad_omega) =#
    fill!(grad_f.grad_omega, 0)
    map!(abs, view(f.m_work, :,  1, :), f.rho_hat)
    map!(conj, f.rho_hat, f.rho_hat)
    map!(/, f.rho_hat, f.rho_hat, view(f.m_work, :, 1, :))

    for b in 1:f.n_b
        rmul!(view(f.rho_hat, :, b), f.w_list[b])
    end

    Threads.@threads for k in 1:f.n_k
        for b in 1:f.n_b
            #= ] = f.rho_hat[:, b] * f.w_list[b] =#
            for n = 1:f.n_e
                axpy!(f.rho_hat[n, b], view(f.r, :, n, k, b), view(grad_f.grad_omega, :, n, k))
            end
        end
    end
    #= LinearAlgebra.axpy!((-2 / f.n_k), grad_f.grad_omega, grad_f.grad_omega) =#

    #= SCDM.project!(UTensor, grad_f.grad_omega, grad_f.grad_work, f.n_k, f.n_e) =#
    Threads.@threads for k in 1:f.n_k
        lmul!(-2 / f.n_k, view(grad_f.grad_omega, :, :, k))
        #= LinearAlgebra.BLAS.gemm!('C', 'N', ComplexF64(1), view(UTensor, :, :, k),
            view(grad_f.grad_omega, :, :, k), ComplexF64(0), grad_f.grad_work)
        grad_f.grad_omega[:, :, k] = grad_f.grad_work - grad_f.grad_work' =#

        #= BLAS.gemm!('C', 'N', ComplexF64(1), view(u, :, :, k), view(grad_f.grad_omega, :, :, k), ComplexF64(0), view(f.r, :, :, k, 1))
        mul!(view(grad_f.grad_omega, :, :, k), view(u, :, :, k), anti_symmetrize!(view(f.r, :, :, k, 2), view(f.r, :, :, k, 1))) =#
        thread_work = zeros(ComplexF64, f.n_e, f.n_e)

        BLAS.gemm!('C', 'N', ComplexF64(1), view(u, :, :, k), view(grad_f.grad_omega, :, :, k), ComplexF64(0), thread_work)
        mul!(view(grad_f.grad_omega, :, :, k), view(u, :, :, k), (thread_work - thread_work') / 2)
        #= mul!(view(grad_f.grad_omega, :, :, k), view(u, :, :, k), (view(f.r, :, :, k, 1) - view(f.r, :, :, k, 1)')/2) =#
    end
    return grad_f.grad_omega
end

struct SVDRetraction end
struct QRRetraction end
struct ExpRetraction end

function retract!(u_buffer, u, d_u, t, ::QRRetraction)
    n_e, _, n_k = size(u)
    copy!(u_buffer, u)
    Threads.@threads for k in 1:n_k
        axpy!(t, view(d_u, :, :, k), view(u_buffer, :, :, k))
        for p = 1:n_e
            norm_p = norm(view(u_buffer, :, p, k))
            lmul!(1 / norm_p, view(u_buffer, :, p, k))
            for q = p+1:n_e
                r = dot(view(u_buffer, :, p, k), view(u_buffer, :, q, k))
                axpy!(-r, view(u_buffer, :, p, k), view(u_buffer, :, q, k))
            end
        end
    end
end

function retract!(u_buffer, u, d_u, t, ::SVDRetraction)
    Nk = size(u, 3)
    copy!(u_buffer, u)
    axpy!(t, d_u, u_buffer)
    for k in 1:Nk
        U, _, V = svd(view(u_buffer, :, :, k))
        u_buffer[:, :, k] = U * V'
    end
end

function retract!(u_buffer, u, d_u, t, ::ExpRetraction)
    Nk = size(u, 3)

    for k in 1:Nk
        #= u_buffer[:, :, k] = u_tensor[:, :, k] * exp(t * Hermitian(u_tensor[:, :, k]' * p_curr[:, :, k])) =#
        work = u[:, :, k]' * d_u[:, :, k]
        u_buffer[:, :, k] = u[:, :, k] * exp(t * (work - work') / 2)
    end
end

