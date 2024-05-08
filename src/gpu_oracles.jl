using CUDA
export GPUOracleF, GPUOracleGradF, make_f_gpu, make_grad_f_gpu, retract_gpu!
using CUDA: blockidx

struct GPUOracleF
    s::CuArray{ComplexF64,4}
    w_list::Vector{Float64}
    k_plus_b::Matrix{Int64}
    k_plus_b_gpu::CuArray{Int64,2}
    n_k::Int64
    n_b::Int64
    n_e::Int64
    n_j::Int64
    r::CuArray{ComplexF64,4}
    rho_hat::Matrix{ComplexF64}
    omega::Vector{Float64}
    m_work::Matrix{ComplexF64}
    w_list_gpu::CuArray{Float64}
    rho_hat_gpu::CuArray{ComplexF64,2}
    omega_gpu::CuArray{Float64,1}
    m_work_gpu::CuArray{ComplexF64,2}
end

function make_f_gpu(s, w_list, k_plus_b, n_k, n_b, n_e, n_j)
    r = CUDA.zeros(ComplexF64, size(s))
    rho_hat = CUDA.zeros(ComplexF64, n_e, n_b)
    omega = zeros(Float64, 1)
    m_work = CUDA.zeros(ComplexF64, n_e, n_b)
    GPUOracleF(s, w_list, k_plus_b, CuArray(k_plus_b), n_k, n_b, n_e, n_j, r,
        rho_hat, omega, m_work,
        CuArray(w_list),
        CuArray(rho_hat), CUDA.zeros(ComplexF64, n_b), CuArray(m_work)
    )
end

function gpu_kmul!(r, s, u, k_plus_b, n_k, n_e)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(k_plus_b)
        b = div(i - 1, n_k) + 1
        k = mod(i - 1, n_k) + 1
        for p in 1:n_e
            for q in 1:n_e
                r[p, q, k, b] = 0
            end
        end
        for z in 1:n_e
            for q in 1:n_e
                for p in 1:n_e
                    r[p, q, k, b] += s[p, z, k, b] * u[z, q, k_plus_b[k, b]]
                end
            end
        end
    end
end

#= function gpu_diag_prod!(m_work, r, u, n_k, n_b, n_e)

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:n_k*n_b
        b = div(i - 1, n_k) + 1
        k = mod(i - 1, n_k) + 1

        for n in 1:n_e
            for p in 1:n_e
                m_work[n, b] += conj(u[p, n, k]) * r[p, n, k, b]
            end
        end
    end
end =#

function (f::GPUOracleF)(u::CuArray{ComplexF64,3})
    fill!(f.omega, 0)
    fill!(f.m_work, 0)
    fill!(f.rho_hat, 0)

    fill!(f.m_work_gpu, 0)
    fill!(f.rho_hat_gpu, 0)
    #= for b in 1:f.n_b
        for k in 1:f.n_k
            mul!(view(f.r, :, :, k, b), view(f.s, :, :, k, b), view(u, :, :, f.k_plus_b[k, b]))
        end
    end =#

    #= for i = 1:length(f.k_plus_b)
        b = div(i - 1, f.n_k) + 1
        k = mod(i - 1, f.n_k) + 1
        mul!(view(f.r, :, :, k, b), view(f.s, :, :, k, b), view(u, :, :, f.k_plus_b[k, b]))
    end =#
    t_1 = time_ns()
    @cuda threads = 256 blocks = 1 gpu_kmul!(f.r, f.s, u, f.k_plus_b_gpu, f.n_k, f.n_e)
    t_2 = time_ns()

    for b in 1:f.n_b
        for k in 1:f.n_k
            #= for n in 1:f.n_e
                f.m_work[n, b] += dot(view(u, :, n, k), view(f.r, :, n, k, b))
            end =#
            Base.mapreduce(*, +, view(u, :, :, k), view(f.r, :, :, k, b), dims=1)
            sum(broadcast!(*, view(u, :, :, k), view(f.r, :, :, k, b)); dims=1)
        end
    end

    #= @cuda threads = 2 blocks = 1 gpu_diag_prod!(f.m_work_gpu, f.r, u, f.n_k, f.n_b, f.n_e) =#

    axpy!(1 / f.n_k, f.m_work_gpu, f.rho_hat_gpu)
    #=
        fill!(f.omega_gpu, f.n_e)
        broadcast!(-, f.omega_gpu, f.omega_gpu,
            vec(mapreduce(abs, +, f.rho_hat_gpu; dims=1)))
        broadcast!(*, f.omega_gpu, f.omega_gpu, f.w_list_gpu)
        t_3 = time_ns()
        f.omega[1] = 2 * sum(f.omega_gpu)
        t_4 = time_ns()

        println((t_2 - t_1) / 1e6, " ms")
        println((t_3 - t_2) / 1e6, " ms")
        println((t_4 - t_3) / 1e6, " ms") =#

    for b in 1:f.n_b
        f.omega[1] = f.omega[1] + 2 * f.w_list[b] * (f.n_e - sum(abs.(view(f.rho_hat_gpu, :, b))))
    end
    #= for b in 1:f.n_b
        axpy!(1/f.n_k, view(f.m_work, :, b), view(f.rho_hat, :, b))
        #= f.rho_hat[:, b] = f.m_work[:, b] / f.n_k =#
        f.omega[1] = f.omega[1] + 2 * f.w_list[b] * (f.n_e - sum(abs.(view(f.rho_hat, :, b))))
    end =#

    return first(f.omega)
end

struct GPUOracleGradF
    f::GPUOracleF
    grad_work::CuArray{ComplexF64,2}
    grad_omega::CuArray{ComplexF64,3}
end

function make_grad_f_gpu(f::GPUOracleF)
    grad_work = CUDA.zeros(ComplexF64, f.n_e, f.n_e)
    grad_omega = CUDA.zeros(ComplexF64, f.n_e, f.n_e, f.n_k)
    return GPUOracleGradF(f, grad_work, grad_omega)
end

function (grad_f::GPUOracleGradF)(u::CuArray{ComplexF64,3})
    f = grad_f.f
    #= grad_f_oracle!(f.r, f.w_list, f.rho_hat, f.n_k, f.n_b, f.n_e, f.Nj, grad_f.grad_omega) =#
    fill!(grad_f.grad_omega, 0)
    copy!(f.rho_hat, conj.(f.rho_hat) ./ abs.(f.rho_hat))

    for b in 1:f.n_b
        rmul!(view(f.rho_hat, :, b), f.w_list[b])
        #= ] = f.rho_hat[:, b] * f.w_list[b] =#
        for k in 1:f.n_k
            for n = 1:f.n_e
                axpy!(f.rho_hat[n, b], view(f.r, :, n, k, b), view(grad_f.grad_omega, :, n, k))
            end
        end
    end
    #= LinearAlgebra.axpy!((-2 / f.n_k), grad_f.grad_omega, grad_f.grad_omega) =#
    lmul!(-2 / f.n_k, grad_f.grad_omega)

    #= SCDM.project!(UTensor, grad_f.grad_omega, grad_f.grad_work, f.n_k, f.n_e) =#
    for k in 1:f.n_k
        #= LinearAlgebra.BLAS.gemm!('C', 'N', ComplexF64(1), view(UTensor, :, :, k),
            view(grad_f.grad_omega, :, :, k), ComplexF64(0), grad_f.grad_work)
        grad_f.grad_omega[:, :, k] = grad_f.grad_work - grad_f.grad_work' =#

        CUDA.CUBLAS.gemm!('C', 'N', ComplexF64(1), view(u, :, :, k), view(grad_f.grad_omega, :, :, k), ComplexF64(0), grad_f.grad_work)
        mul!(view(grad_f.grad_omega, :, :, k), view(u, :, :, k), (grad_f.grad_work - grad_f.grad_work') / 2)
    end
    return grad_f.grad_omega
end

function retract_gpu!(u_buffer, u, d_u, t, ::QRRetraction)
    Nk = size(u, 3)
    copy!(u_buffer, u)
    axpy!(t, d_u, u_buffer)
    #= time_1 = time_ns() =#
    qrfacs = [qr!(u_buffer[:, :, k]) for k = 1:Nk]
    #= time_2 = time_ns() =#
    for k in 1:Nk
        #= qrfac = qr!(u_buffer[:, :, k]) =#
        d = diag(qrfacs[k].R)
        D = Diagonal(sign.(sign.(d .+ 1 // 2)))
        Q = Matrix(qrfacs[k].Q)
        CUDA.@allowscalar mul!(view(u_buffer, :, :, k), Q, D)
    end
    #= time_3 = time_ns()
    println("qr time: ", time_2 - time_1)
    println("mul time: ", time_3 - time_2) =#
end

function retract_gpu!(u_buffer, u, d_u, t, ::SVDRetraction)
    Nk = size(u, 3)
    for k in 1:Nk
        u_buffer[:, :, k] = u[:, :, k] + t * d_u[:, :, k]
        u, _, v = svd(u_buffer[:, :, k])
        u_buffer[:, :, k] = u * v'
    end
end

function retract_gpu!(u_buffer, u, d_u, t, ::ExpRetraction)
    Nk = size(u, 3)

    for k in 1:Nk
        #= u_buffer[:, :, k] = u_tensor[:, :, k] * exp(t * Hermitian(u_tensor[:, :, k]' * p_curr[:, :, k])) =#
        work = u[:, :, k]' * d_u[:, :, k]
        u_buffer[:, :, k] = u[:, :, k] * exp(t * (work - work') / 2)
    end
end

