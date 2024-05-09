using LinearAlgebra
using CUDA
export GPUOracleF, GPUOracleGradF, make_f_gpu, make_grad_f_gpu, retract_gpu!

struct GPUOracleF
    s::CuArray{ComplexF64,4}
    w_list::Vector{Float64}
    k_plus_b::CuArray{Int64,2}
    n_k::Int64
    n_b::Int64
    n_e::Int64
    n_j::Int64
    r::CuArray{ComplexF64,4}
    rho_hat::CuArray{ComplexF64,2}
    omega::Vector{Float64}
    m_work::CuArray{ComplexF64,3}
end

function make_f_gpu(s, w_list, kplusb, n_k, n_b, n_e, n_j)
    r = CUDA.zeros(ComplexF64, size(s))
    rho_hat = CUDA.zeros(ComplexF64, n_e, n_b)
    omega = zeros(Float64, n_b)
    m_work = CUDA.zeros(ComplexF64, n_e, n_k, n_b)
    GPUOracleF(s, w_list, kplusb, n_k, n_b, n_e, n_j, r, rho_hat, omega, m_work)
end

function f_kernel_1(s, r, u, k_plus_b, n_k, n_b, n_e)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:(n_k*n_e)
        k = div(i - 1, n_e) + 1
        p = mod(i - 1, n_e) + 1
        #= end
        for k in index:stride:n_k
            for p in 1:n_e =#
        for z in 1:n_e
            for b in 1:n_b
                for q in 1:n_e
                    r[p, q, k, b] += s[p, z, k, b] * u[z, q, k_plus_b[k, b]]
                end
            end
            #= end =#
        end
    end

end

function f_kernel_2(r, u, m_work, n_k, n_b, n_e)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    for i in index:stride:(n_k*n_e)
        k = div(i - 1, n_e) + 1
        n = mod(i - 1, n_e) + 1
        #= for k in index:stride:n_k
            for n in 1:n_e =#
        for b in 1:n_b
            for i in 1:n_e
                m_work[n, k, b] += u[i, n, k]' * r[i, n, k, b]
            end
        end
    end
    #= end =#
end

#= function f_kernel_3(rho_hat, m_work, n_k, n_b, n_e)
    for n in 1:n_e
        for b in 1:n_b
            for k in 1:n_k
                rho_hat[n, b] += m_work[n, k, b]
            end
        end
    end
end =#

function f_kernel_sum(data)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    s = 1
    while s < length(data)
        for i in index:stride:div(length(data) + 1, 2)
            idx = (i - 1) * 2s + 1
            if idx + s <= length(data)
                data[idx] += data[idx+s]
            end
        end
        sync_threads()
        s *= 2
    end
end

function f_kernel_3(data, n_k, n_b, n_e)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    s = 1
    n_be = n_b * n_e
    k_stride = div(stride, n_be)
    if index > k_stride * n_be
        return 
    end
    k_half = div(n_k + 1, 2)
    k_index = div(index - 1, n_be) + 1
    bn = rem(index-1, n_be) + 1
    b = div(bn-1, n_e) + 1
    n = rem(index-1, n_e) + 1

    while s < n_k
        for i in k_index:k_stride:k_half
            idx = (i - 1) * 2s + 1
            if idx + s <= n_k
                data[n, idx, b] += data[n, idx+s, b]
            end
        end
        sync_threads()
        s *= 2
    end
end


function (f::GPUOracleF)(u)
    fill!(f.rho_hat, 0)
    fill!(f.m_work, 0)
    fill!(f.r, 0)
    @time @cuda threads = 256 f_kernel_1(f.s, f.r, u, f.k_plus_b, f.n_k, f.n_b, f.n_e)
    @time @cuda threads = 256 f_kernel_2(f.r, u, f.m_work, f.n_k, f.n_b, f.n_e)
    #= for k in 1:f.n_k
        for b in 1:f.n_b
            mul!(view(f.r, :, :, k, b), view(f.s, :, :, k, b), view(u, :, :, f.k_plus_b[k, b]))
        end
        for n in 1:f.n_e
            for b in 1:f.n_b
                f.m_work[n, k, b] = dot(view(u, :, n, k), view(f.r, :, n, k, b))
            end
        end
    end =#

    #= p = CUDA.ones(Float64, 27)
    @cuda f_kernel_sum(p)
    println(p) =#

    #= @time for n in 1:f.n_e
        for b in 1:f.n_b
            @cuda f_kernel_sum(view(f.m_work, n, :, b))
        end
    end
    copyto!(f.rho_hat, view(f.m_work, :, 1, :)) =#
    #
    #= @time copyto!(f.rho_hat, sum(f.m_work, dims=2)) =#
    @time @cuda threads=256 f_kernel_3(f.m_work, f.n_k, f.n_b, f.n_e)
    copyto!(f.rho_hat, view(f.m_work, :, 1, :))

    #= for n in 1:f.n_e
        for k in 1:f.n_k
            for b in 1:f.n_b
                f.rho_hat[n, b] += f.m_work[n, k, b]
            end
        end
    end =#
    @time lmul!(1 / f.n_k, f.rho_hat)

    @time for b in 1:f.n_b
        #= f.omega[b] = 2 * f.w_list[b] * (f.n_e - sum(real, view(f.m_work, :, b))) =#
        f.omega[b] = 2 * f.w_list[b] * (f.n_e - sum(abs.(view(f.rho_hat, :, b))))
    end

    return sum(f.omega)
end

struct GPUOracleGradF
    f::GPUOracleF
    grad_work::Matrix{ComplexF64}
    grad_omega::Array{ComplexF64,3}
end

function make_grad_f_gpu(f::GPUOracleF)
    grad_work = zeros(ComplexF64, f.n_e, f.n_e)
    grad_omega = zeros(ComplexF64, f.n_e, f.n_e, f.n_k)
    return GPUOracleGradF(f, grad_work, grad_omega)
end

function anti_symmetrize_gpu!(dst, src)
    for i in axes(src, 1)
        for j in axes(src, 2)
            dst[i, j] = (src[i, j] - src[j, i]') / 2
        end
    end
    return dst
end

function (grad_f::GPUOracleGradF)(u::Array{ComplexF64,3})
    f = grad_f.f
    #= grad_f_oracle!(f.r, f.w_list, f.rho_hat, f.n_k, f.n_b, f.n_e, f.Nj, grad_f.grad_omega) =#
    fill!(grad_f.grad_omega, 0)
    map!(abs, view(f.m_work, :, :, 1), f.rho_hat)
    map!(conj, f.rho_hat, f.rho_hat)
    map!(/, f.rho_hat, f.rho_hat, view(f.m_work, :, :, 1))

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

        BLAS.gemm!('C', 'N', ComplexF64(1), view(u, :, :, k), view(grad_f.grad_omega, :, :, k), ComplexF64(0), view(f.r, :, :, k, 1))
        mul!(view(grad_f.grad_omega, :, :, k), view(u, :, :, k), anti_symmetrize!(view(f.r, :, :, k, 2), view(f.r, :, :, k, 1)))
        #= mul!(view(grad_f.grad_omega, :, :, k), view(u, :, :, k), (view(f.r, :, :, k, 1) - view(f.r, :, :, k, 1)')/2) =#
    end
    return grad_f.grad_omega
end

function retract_gpu!(u_buffer, u, d_u, t, ::QRRetraction)
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

function retract_gpu!(u_buffer, u, d_u, t, ::SVDRetraction)
    Nk = size(u, 3)
    copy!(u_buffer, u)
    axpy!(t, d_u, u_buffer)
    for k in 1:Nk
        U, _, V = svd(view(u_buffer, :, :, k))
        u_buffer[:, :, k] = U * V'
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

