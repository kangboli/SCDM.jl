export OracleF, OracleGradF, make_f, make_grad_f, retract!

ideg = 4
struct OracleF
    STensor::Array{ComplexF64,4}
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
end

function (f::OracleF)(UTensor::Array{ComplexF64,3})
    #= f_oracle!(f.STensor, f.R, UTensor, f.w_list, f.kplusb, f.rho_hat, f.Nk, f.Nb, f.Ne, f.Nj, f.omega) =#

    m_work = zeros(ComplexF64, f.Ne)
    fill!(f.omega, 0)

    for b in 1:f.Nb
        fill!(m_work, 0)
        for k in 1:f.Nk
            #= LinearAlgebra.BLAS.gemm!('N', 'N', one, f.STensor[:, :, k, b], UTensor[:, :, f.kplusb[k, b]], zero, view(f.R, :, :, k, b)) =#
            LinearAlgebra.mul!(view(f.R, :, :, k, b), view(f.STensor, :, :, k, b), view(UTensor, :, :, f.kplusb[k, b]))
            for n in 1:f.Ne
                m_work[n] += dot(UTensor[:, n, k], f.R[:, n, k, b])
            end
        end
        f.rho_hat[:, b] = m_work / f.Nk
        f.omega[1] = f.omega[1] + 2 * f.w_list[b] * (f.Ne - sum(abs.(f.rho_hat[:, b])))
    end

    return first(f.omega)
end

struct OracleGradF
    f::OracleF
    grad_work::Matrix{ComplexF64}
    grad_omega::Array{ComplexF64,3}
end

function make_grad_f(f::OracleF)
    grad_work = zeros(ComplexF64, f.Ne, f.Ne)
    grad_omega = zeros(ComplexF64, f.Ne, f.Ne, f.Nk)
    return OracleGradF(f, grad_work, grad_omega)
end

function (grad_f::OracleGradF)(UTensor::Array{ComplexF64,3})
    f = grad_f.f
    #= grad_f_oracle!(f.R, f.w_list, f.rho_hat, f.Nk, f.Nb, f.Ne, f.Nj, grad_f.grad_omega) =#
    fill!(grad_f.grad_omega, 0)
    copy!(f.rho_hat, conj.(f.rho_hat) ./ abs.(f.rho_hat))

    for b in 1:f.Nb
        rmul!(view(f.rho_hat, :, b), f.w_list[b])
        #= ] = f.rho_hat[:, b] * f.w_list[b] =#
        for k in 1:f.Nk
            for n = 1:f.Ne
                LinearAlgebra.axpy!(f.rho_hat[n, b], view(f.R, :, n, k, b), view(grad_f.grad_omega, :, n, k))
            end
        end
    end
    #= LinearAlgebra.axpy!((-2 / f.Nk), grad_f.grad_omega, grad_f.grad_omega) =#
    lmul!(-2 / f.Nk, grad_f.grad_omega)

    #= SCDM.project!(UTensor, grad_f.grad_omega, grad_f.grad_work, f.Nk, f.Ne) =#
    for k in 1:f.Nk
        #= LinearAlgebra.BLAS.gemm!('C', 'N', ComplexF64(1), view(UTensor, :, :, k),
            view(grad_f.grad_omega, :, :, k), ComplexF64(0), grad_f.grad_work)
        grad_f.grad_omega[:, :, k] = grad_f.grad_work - grad_f.grad_work' =#

        LinearAlgebra.BLAS.gemm!('C', 'N', ComplexF64(1), view(UTensor, :, :, k), view(grad_f.grad_omega, :, :, k), ComplexF64(0), grad_f.grad_work)
        mul!(view(grad_f.grad_omega, :, :, k), view(UTensor, :, :, k), (grad_f.grad_work - grad_f.grad_work')/2)
        #= grad_f.grad_omega[:, :, k] = UTensor[:, :, k] * (grad_f.grad_work - grad_f.grad_work')/2 =#
        #= grad_f.grad_omega[:, :, k] = (work - work')/2 =#
        #= grad_f.grad_omega[:, :, k] = grad_f.grad_omega[:, :, k] - UTensor[:, :, k] * (work + work') / 2 =#

        #= work = UTensor[:, :, k]'  * grad_f.grad_omega[:, :, k]
        grad_f.grad_omega[:, :, k] = UTensor[:, :, k] * (work - work') =#
    end
    return grad_f.grad_omega
end

struct SVDRetraction end
struct QRRetraction end
struct ExpRetraction end

function retract!(u_buffer, u_tensor, p_curr, t, ::QRRetraction)
    Nk = size(u_tensor, 3)
    for k in 1:Nk
        #= u_buffer[:, :, k] = u_tensor[:, :, k] + t * p_curr[:, :, k] =#
        copy!(view(u_buffer, :, :, k), view(u_tensor, :, :, k))
        axpy!(t, view(p_curr, :, :, k), view(u_buffer, :, :, k))
        qrfac = qr!(u_buffer[:, :, k])
        d = diag(qrfac.R)
        D = Diagonal(sign.(sign.(d .+ 1 // 2)))
        mul!(view(u_buffer, :, :, k), Matrix(qrfac.Q), D)
    end
end

function retract!(u_buffer, u_tensor, p_curr, t, ::SVDRetraction)
    Nk = size(u_tensor, 3)
    for k in 1:Nk
        u_buffer[:, :, k] = u_tensor[:, :, k] + t * p_curr[:, :, k]
        u, _, v = svd(u_buffer[:, :, k])
        u_buffer[:, :, k] = u * v'
    end
end

function retract!(u_buffer, u_tensor, p_curr, t, ::ExpRetraction)
    Nk = size(u_tensor, 3)

    for k in 1:Nk
        #= u_buffer[:, :, k] = u_tensor[:, :, k] * exp(t * Hermitian(u_tensor[:, :, k]' * p_curr[:, :, k])) =#
        work = u_tensor[:, :, k]' * p_curr[:, :, k]
        u_buffer[:, :, k] = u_tensor[:, :, k] * exp(t *  (work - work')/2)
    end
end

