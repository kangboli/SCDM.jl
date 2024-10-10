export OracleF, OracleGradF, make_f, make_grad_f, qr_retract!, svd_retract!, exp_retract!, peek_centers, peek_spreads, random_gauge, as_mv

mutable struct OracleMVF <: AbstractOracleF
    s::Array{ComplexF64,4}
    w_list::Vector{Float64}
    shell_list::Matrix{Float64}
    k_plus_b::Matrix{Int64}
    k_minus_b::Matrix{Int64}
    n_k::Int64
    n_b::Int64
    n_e::Int64
    n_j::Int64
    r::Array{ComplexF64,4}
    rho_hat::Matrix{ComplexF64}
    omega::Vector{Float64}
    m_work::Array{ComplexF64,3}
    gradient_ready::Bool
end

as_mv(f::OracleF) = OracleMVF(f.s, f.w_list, f.shell_list, f.k_plus_b, f.k_minus_b, f.n_k, f.n_b, f.n_e, f.n_j, f.r, f.rho_hat, f.omega, f.m_work, false)

function (f::OracleMVF)(u::Array{ComplexF64,3})
    fill!(f.rho_hat, 0)
    fill!(f.m_work, 0)
    for k in 1:f.n_k
        for b in 1:f.n_b
            mul!(view(f.r, :, :, k, b), view(f.s, :, :, k, b), view(u, :, :, f.k_plus_b[k, b]))
        end
        for n in 1:f.n_e
            for b in 1:f.n_b
                f.m_work[n, k, b] = dot(view(u, :, n, k), view(f.r, :, n, k, b))
            end
        end
    end

    for n in 1:f.n_e
        for k in 1:f.n_k
            for b in 1:f.n_b
                f.rho_hat[n, b] += f.m_work[n, k, b]
            end
        end
    end
    lmul!(1 / f.n_k, f.rho_hat)

    for b in 1:f.n_b
        #= f.omega[b] = 2 * f.w_list[b] * (f.n_e - sum(real, view(f.m_work, :, b))) =#
        f.omega[b] = f.w_list[b] * (f.n_e - sum(abs2.(view(f.rho_hat, :, b))))
    end

    f.gradient_ready = true
    return sum(f.omega)
end

function peek_centers(f::OracleMVF)
    phase = dropdims(sum(f.m_work, dims=2), dims=2)
    lmul!(1 / f.n_k, phase)
    map!(angle, phase, phase)

    centers = zeros(Float64, 3, f.n_e)
    for n in 1:f.n_e
        for b in 1:f.n_b
            centers[:, n] += f.w_list[b] * f.shell_list[:, b] * phase[n, b]
        end
    end
    return centers
end

function peek_spreads(f::OracleMVF)
    spreads = zeros(Float64, f.n_e)

    for n in 1:f.n_e
        for b in 1:f.n_b
            #= f.omega[b] = 2 * f.w_list[b] * (f.n_e - sum(real, view(f.m_work, :, b))) =#
            spreads[n] += 2 * f.w_list[b] * (1 - abs.(view(f.rho_hat, n, b)))
        end
    end
    return spreads
end

struct OracleGradMVF
    f::OracleMVF
    grad_work::Matrix{ComplexF64}
    grad_omega::Array{ComplexF64,3}
end

function make_grad_f(f::OracleMVF)
    grad_work = zeros(ComplexF64, f.n_e, f.n_e)
    grad_omega = zeros(ComplexF64, f.n_e, f.n_e, f.n_k)
    return OracleGradMVF(f, grad_work, grad_omega)
end

function anti_symmetrize!(dst, src)
    for i in axes(src, 1)
        for j in axes(src, 2)
            dst[i, j] = (src[i, j] - src[j, i]') / 2
        end
    end
    return dst
end

function (grad_f::OracleGradMVF)(u::Array{ComplexF64,3})
    f = grad_f.f
    f.gradient_ready || error("the oracle was not first evaluated")
    f.gradient_ready = false
    #= grad_f_oracle!(f.r, f.w_list, f.rho_hat, f.n_k, f.n_b, f.n_e, f.Nj, grad_f.grad_omega) =#
    fill!(grad_f.grad_omega, 0)
    map!(abs, view(f.m_work, :, 1, :), f.rho_hat)
    map!(conj, f.rho_hat, f.rho_hat)
    # map!(/, f.rho_hat, f.rho_hat, view(f.m_work, :, 1, :))

    for b in 1:f.n_b
        rmul!(view(f.rho_hat, :, b), f.w_list[b])
    end

    for k in 1:f.n_k
        for b in 1:f.n_b
            #= ] = f.rho_hat[:, b] * f.w_list[b] =#
            for n = 1:f.n_e
                axpy!(f.rho_hat[n, b], view(f.r, :, n, k, b), view(grad_f.grad_omega, :, n, k))
            end
        end
    end
    #= LinearAlgebra.axpy!((-2 / f.n_k), grad_f.grad_omega, grad_f.grad_omega) =#

    #= SCDM.project!(UTensor, grad_f.grad_omega, grad_f.grad_work, f.n_k, f.n_e) =#
    for k in 1:f.n_k
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

