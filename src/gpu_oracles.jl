
export OracleF, OracleGradF, make_f, make_grad_f, retract!

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
    #= f_oracle!(f.STensor, f.r, UTensor, f.w_list, f.kplusb, f.rho_hat, f.n_k, f.n_b, f.n_e, f.Nj, f.omega) =#
    #= LinearAlgebra.BLAS.gemm!('N', 'N', one, f.STensor[:, :, k, b], UTensor[:, :, f.kplusb[k, b]], zero, view(f.r, :, :, k, b)) =#
    #= rho_hat_handle = reinterpret(Float64, f.rho_hat) =#
    fill!(f.rho_hat, 0)
    fill!(f.m_work, 0)
    fill!(f.r, 0)
    BLAS.set_num_threads(1)
    @time Threads.@threads for k in 1:f.n_k
        #= @time for k in 1:f.n_k =#
        for b in 1:f.n_b
            #= for z in 1:f.n_e
                for p in 1:f.n_e
                    for q in 1:f.n_e
                        @inbounds f.r[p, q, k, b] += f.s[p, z, k, b] * u[z, q, f.k_plus_b[k, b]]
                    end
                end
            end =#
            @inbounds mul!(view(f.r, :, :, k, b), view(f.s, :, :, k, b), view(u, :, :, f.k_plus_b[k, b]))
        end
    end
    BLAS.set_num_threads(Threads.nthreads)

    for k in 1:f.n_k
        for n in 1:f.n_e
            for b in 1:f.n_b
                @inbounds f.m_work[n, k, b] = dot(view(u, :, n, k), view(f.r, :, n, k, b))
                #= for i in 1:f.n_e
                    @inbounds f.m_work[n, k, b] += conj(u[i, n, k]) * f.r[i, n, k, b]
                end =#
            end
        end
    end

    #= rho_hat_handle = reinterpret(Float64, f.rho_hat)
    m_work_handle = reinterpret(Float64, f.m_work) =#
    for k in 1:f.n_k
        for n in 1:f.n_e
            #= n1, n2 = 2n - 1, 2n =#
            for b in 1:f.n_b
                @inbounds f.rho_hat[n, b] += f.m_work[n, k, b]
                #= @inbounds rho_hat_handle[n1, b] += m_work_handle[n1, b]
                @inbounds rho_hat_handle[n2, b] += m_work_handle[n2, b] =#
            end
        end
    end
    lmul!(1 / f.n_k, f.rho_hat)
    #= map!(abs, view()f.m_work, f.rho_hat) =#
    #= axpby!(1 / f.n_k, f.m_work, 0, f.rho_hat) =#

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

function (grad_f::OracleGradF)(u::Array{ComplexF64,3})
    f = grad_f.f
    #= grad_f_oracle!(f.r, f.w_list, f.rho_hat, f.n_k, f.n_b, f.n_e, f.Nj, grad_f.grad_omega) =#
    fill!(grad_f.grad_omega, 0)
    copy!(f.rho_hat, conj.(f.rho_hat) ./ abs.(f.rho_hat))

    for b in 1:f.n_b
        rmul!(view(f.rho_hat, :, b), f.w_list[b])
        #= ] = f.rho_hat[:, b] * f.w_list[b] =#
        for n = 1:f.n_e
            axpy!(f.rho_hat[n, b], view(f.r, :, n, :, b), view(grad_f.grad_omega, :, n, :))
        end
    end
    #= LinearAlgebra.axpy!((-2 / f.n_k), grad_f.grad_omega, grad_f.grad_omega) =#
    lmul!(-2 / f.n_k, grad_f.grad_omega)

    #= SCDM.project!(UTensor, grad_f.grad_omega, grad_f.grad_work, f.n_k, f.n_e) =#
    for k in 1:f.n_k
        #= LinearAlgebra.BLAS.gemm!('C', 'N', ComplexF64(1), view(UTensor, :, :, k),
            view(grad_f.grad_omega, :, :, k), ComplexF64(0), grad_f.grad_work)
        grad_f.grad_omega[:, :, k] = grad_f.grad_work - grad_f.grad_work' =#

        BLAS.gemm!('C', 'N', ComplexF64(1), view(u, :, :, k), view(grad_f.grad_omega, :, :, k), ComplexF64(0), grad_f.grad_work)
        mul!(view(grad_f.grad_omega, :, :, k), view(u, :, :, k), (grad_f.grad_work - grad_f.grad_work') / 2)
    end
    return grad_f.grad_omega
end

struct SVDRetraction end
struct QRRetraction end
struct ExpRetraction end

function retract!(u_buffer, u, d_u, t, ::QRRetraction)
    n_e, _, n_k = size(u)
    copy!(u_buffer, u)
    axpy!(t, d_u, u_buffer)
    #= time_1 = time_ns() =#
    #= qrfacs = [qr!(u_buffer[:, :, k]) for k = 1:n_k] =#
    #= time_2 = time_ns() =#
    for k in 1:n_k
        for p = 1:n_e
            norm_p = norm(view(u_buffer, :, p, k))
            lmul!(1 / norm_p, view(u_buffer, :, p, k))
            for q = p+1:n_e
                r = dot(view(u_buffer, :, p, k), view(u_buffer, :, q, k))
                axpy!(-r, view(u_buffer, :, p, k), view(u_buffer, :, q, k))
            end
        end
        #= d = diag(qrfacs[k].R)
        D = Diagonal(sign.(sign.(d .+ 1 // 2)))
        mul!(view(u_buffer, :, :, k), Matrix(qrfacs[k].Q), D) =#
    end
    #= time_3 = time_ns()
    println("qr time: ", time_2 - time_1)
    println("mul time: ", time_3 - time_2) =#
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

    #= for n in 1:f.n_e
        for b in 1:f.n_b
            @inbounds f.rho_hat[n, b] = dot(view(u, :, n, :), view(f.r, :, n, :, b))
        end
    end =#
