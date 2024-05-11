using LinearAlgebra
using CUDA
export GPUOracleF, GPUOracleGradF, make_f_gpu, make_grad_f_gpu, retract_gpu!

const ComplexGPU = ComplexF32

struct GPUOracleF
    s::CuArray{ComplexGPU,4}
    w_list::Vector{Float32}
    k_plus_b::CuArray{Int64,2}
    k_minus_b::CuArray{Int64,2}
    n_k::Int32
    n_b::Int32
    n_e::Int32
    n_j::Int32
    r::CuArray{ComplexGPU,4}
    rho_hat_cpu::Array{ComplexGPU,2}
    rho_hat::CuArray{ComplexGPU,2}
    omega::Vector{Float32}
    m_work::CuArray{ComplexGPU,3}
end

function make_f_gpu(s, w_list, k_plus_b, k_minus_b, n_k, n_b, n_e, n_j)
    s = CuArray{ComplexGPU,4}(s)
    r = CUDA.zeros(ComplexGPU, size(s))
    rho_hat_cpu = zeros(ComplexGPU, n_e, n_b)
    rho_hat = CUDA.zeros(ComplexGPU, n_e, n_b)
    omega = zeros(Float64, n_b)
    m_work = CUDA.zeros(ComplexGPU, n_e, n_k, n_b)
    for b in 1:n_b
        for k in 1:n_k
            transpose!(view(r, :, :, k, b), view(s, :, :, k, b))
            copyto!(view(s, :, :, k, b), view(r, :, :, k, b))
        end
    end
    GPUOracleF(s, w_list, k_plus_b, k_minus_b, n_k, n_b, n_e, n_j, r, rho_hat_cpu, rho_hat, omega, m_work)
end

function f_kernel_1(st, r, u, k_plus_b, k_minus_b, n_k, n_b, n_e, n_s)
    #= index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x =#
    #= n_e2 = n_e^2 =#
    u_tmp = CuDynamicSharedArray(ComplexGPU, n_e * n_s)
    #= for i in index-1:stride:(n_k*n_e2)-1 =#
    #= n_bar = div(n_e - 1, n_s) + 1 =#
    bar, k = divrem(blockIdx().x - 1, n_k)
    k += 1
    bar += 1
    pq = threadIdx().x
    #= k, pq = divrem(i, n_e2) =#
    # pq, k = divrem(i, n_k)
    #= k += 1
    pq += 1 =#
    q, p = divrem(pq - 1, n_e)
    #= p, q  = divrem(pq - 1, n_s) =#
    p += 1
    q += 1
    qb = q + (bar - 1) * n_s
    if qb > n_e
        return nothing
    end

    u_tmp[pq] = u[p, qb, k]
    sync_threads()
    #= end
    for k in index:stride:n_k
        for p in 1:n_e =#
    for b in 1:n_b
        # kpb = k_plus_b[k, b]
        kmb = k_minus_b[k, b]
        qe = (q - 1) * n_e
        for z in 1:n_e
            #= for q in 1:n_e =#
            # r[p, q, k, b] += s[p, z, k, b] * u[z, q, kpb]
            #= r[p, q, kmb, b] += st[z, p, kmb, b] * u[z, q, k] =#
            #= r[p, qb, kmb, b] += st[z, p, kmb, b] * u[z, qb, k] =#
            r[p, qb, kmb, b] += st[z, p, kmb, b] * u_tmp[z+qe]
        end
        #= end =#
        #= end =#
    end
    #= end =#

end

function f_kernel_2(r, u, m_work, n_k, n_b, n_e)
    #= index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x =#
    k = blockIdx().x
    n = threadIdx().x
    #= for i in index:stride:(n_k*n_e)
        k, n = divrem(i - 1, n_e)
        k += 1
        n += 1 =#
    #= for k in index:stride:n_k
        for n in 1:n_e =#
    for b in 1:n_b
        for i in 1:n_e
            m_work[n, k, b] += u[i, n, k]' * r[i, n, k, b]
        end
    end
    #= end =#
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

function f_kernel_3(rho_hat, m_work, n_k, n_b, n_e)
    index = threadIdx().x
    stride = blockDim().x
    b, n = divrem(blockIdx().x - 1, n_e)
    b += 1
    n += 1
    s = 1
    while s < n_k
        for i in index:stride:div(n_k + 1, 2)
            idx = (i - 1) * 2s + 1
            if idx + s <= n_k
                m_work[n, idx, b] += m_work[n, idx+s, b]
            end
        end
        sync_threads()
        s *= 2
    end

    sync_threads()
    if index == 1
        rho_hat[n, b] = m_work[n, 1, b]
    end
    nothing
end

#= function f_kernel_3(rho_hat, m_work, n_k, n_b, n_e)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    s = 1
    n_be = n_b * n_e
    k_stride = div(stride, n_be)
    if index > k_stride * n_be
        return nothing
    end
    k_half = div(n_k + 1, 2)
    k_index = div(index - 1, n_be) + 1
    bn = rem(index - 1, n_be) + 1
    b = div(bn - 1, n_e) + 1
    n = rem(index - 1, n_e) + 1

    while s < n_k
        for i in k_index:k_stride:k_half
            idx = (i - 1) * 2s + 1
            if idx + s <= n_k
                m_work[n, idx, b] += m_work[n, idx+s, b]
            end
        end
        sync_threads()
        s *= 2
    end

    sync_threads()
    if k_index == 1
        rho_hat[n, b] = m_work[n, 1, b]
    end
    return nothing
end
 =#

function (f::GPUOracleF)(u)
    fill!(f.rho_hat, 0)
    fill!(f.m_work, 0)
    fill!(f.r, 0)
    n_s = f.n_e
    n_bar = 1
    CUDA.@sync begin
        #= @cuda threads = f.n_e * f.n_e blocks = f.n_k shmem = sizeof(ComplexGPU) * f.n_e * f.n_e f_kernel_1(f.s, f.r, u, f.k_plus_b, f.k_minus_b, f.n_k, f.n_b, f.n_e, f.n_e) =#
        @cuda threads = f.n_e * n_s blocks = f.n_k * n_bar shmem = sizeof(ComplexGPU) * f.n_e * n_s f_kernel_1(f.s, f.r, u, f.k_plus_b, f.k_minus_b, f.n_k, f.n_b, f.n_e, n_s)
        @cuda threads = f.n_e blocks = f.n_k f_kernel_2(f.r, u, f.m_work, f.n_k, f.n_b, f.n_e)
        #= @cuda threads = 512 f_kernel_3(f.rho_hat, f.m_work, f.n_k, f.n_b, f.n_e) =#
        @cuda threads = min(f.n_k, 512) blocks = f.n_e * f.n_b f_kernel_3(f.rho_hat, f.m_work, f.n_k, f.n_b, f.n_e)
    end
    #= @time copyto!(f.rho_hat_cpu, view(f.m_work, :, 1, :)) =#
    copyto!(f.rho_hat_cpu, f.rho_hat)

    #= for n in 1:f.n_e
        for k in 1:f.n_k
            for b in 1:f.n_b
                f.rho_hat[n, b] += f.m_work[n, k, b]
            end
        end
    end =#
    lmul!(1 / f.n_k, f.rho_hat_cpu)

    for b in 1:f.n_b
        #= f.omega[b] = 2 * f.w_list[b] * (f.n_e - sum(real, view(f.m_work, :, b))) =#
        f.omega[b] = 2 * f.w_list[b] * (f.n_e - sum(abs.(view(f.rho_hat_cpu, :, b))))
    end

    return sum(f.omega)
end

struct GPUOracleGradF
    f::GPUOracleF
    grad_work::CuArray{ComplexGPU,3}
    grad_omega::CuArray{ComplexGPU,3}
end

function make_grad_f_gpu(f::GPUOracleF)
    grad_work = CUDA.zeros(ComplexGPU, f.n_e, f.n_e, f.n_k)
    grad_omega = CUDA.zeros(ComplexGPU, f.n_e, f.n_e, f.n_k)
    return GPUOracleGradF(f, grad_work, grad_omega)
end

function anti_symmetrize_gpu!(dst, src)
    i = threadIdx().x
    j = blockIdx().x
    #= for i in axes(src, 1)
        for j in axes(src, 2) =#
    dst[i, j] = (src[i, j] - src[j, i]') / 2
    #= end
    end =#
    return nothing
end


function df_kernel_1(rho_hat, r, grad_omega, n_e, n_b)
    k = blockIdx().x
    n = threadIdx().x
    #= for k in 1:n_k =#
    for b in 1:n_b
        #= ] = f.rho_hat[:, b] * f.w_list[b] =#
        #= for n = 1:n_e =#
        for i = 1:n_e
            grad_omega[i, n, k] += rho_hat[n, b] * r[i, n, k, b]
            #= axpy!(rho_hat[n, b], view(f.r, :, n, k, b), view(grad_f.grad_omega, :, n, k)) =#
        end
    end
    #= end
    end =#
end

function df_kernel_2(grad_work, grad_work_2, grad_omega, u, n_e, n_s, n_k)
    k = blockIdx().x
    ij = threadIdx().x
    i, j = divrem(ij - 1, n_s)
    i += 1
    j += 1
    grad_work[i, j, k] = grad_omega[i, j, k] / 2
    for q in 1:n_e
        grad_work_2[i, j, k] += grad_omega[q, i, k]' * u[q, j, k] / 2
    end

    sync_threads()

    for p in 1:n_e
        grad_work[i, j, k] -= u[i, p, k] * grad_work_2[p, j, k]
    end

    # for q in 1:n_e
    #     for p in 1:n_e
    #         grad_work[i, j, k] -= (u[i, p, k] * grad_omega[q, p, k]' * u[q, j, k]) / 2
    #     end
    # end

    grad_omega[i, j, k] = grad_work[i, j, k] * (-2 / (n_k))
    return nothing
end

function (grad_f::GPUOracleGradF)(u)
    f = grad_f.f
    #= grad_f_oracle!(f.r, f.w_list, f.rho_hat, f.n_k, f.n_b, f.n_e, f.Nj, grad_f.grad_omega) =#
    fill!(grad_f.grad_omega, 0)
    map!(abs, view(f.m_work, :, 1, :), f.rho_hat)
    map!(conj, f.rho_hat, f.rho_hat)
    map!(/, f.rho_hat, f.rho_hat, view(f.m_work, :, 1, :))

    for b in 1:f.n_b
        rmul!(view(f.rho_hat, :, b), f.w_list[b])
    end

    grad_work_2 = similar(grad_f.grad_work)
    fill!(grad_work_2, 0)

    CUDA.@sync begin
        @cuda threads = f.n_e blocks = f.n_k df_kernel_1(f.rho_hat, f.r, grad_f.grad_omega, f.n_e, f.n_b)
        @cuda threads = f.n_e^2 blocks = f.n_k df_kernel_2(grad_f.grad_work, grad_work_2, grad_f.grad_omega, u, f.n_e, f.n_e, f.n_k)
    end
    #= for k in 1:f.n_k
        lmul!(-2 / f.n_k, view(grad_f.grad_omega, :, :, k))
        #= LinearAlgebra.BLAS.gemm!('C', 'N', ComplexF64(1), view(UTensor, :, :, k),
            view(grad_f.grad_omega, :, :, k), ComplexF64(0), grad_f.grad_work)
        grad_f.grad_omega[:, :, k] = grad_f.grad_work - grad_f.grad_work' =#

        #= CUDA.BLAS.gemm!('C', 'N', ComplexGPU(1), view(u, :, :, k), view(grad_f.grad_omega, :, :, k), ComplexGPU(0), view(f.r, :, :, k, 1)) =#
        mul!(view(f.r, :, :, k, 1), view(u, :, :, k)', view(grad_f.grad_omega, :, :, k))
        #= @cuda threads = f.n_e blocks = f.n_e anti_symmetrize_gpu!(view(f.r, :, :, k, 2), view(f.r, :, :, k, 1)) =#
        mul!(view(grad_f.grad_omega, :, :, k), view(u, :, :, k), (view(f.r, :, :, k, 1) - view(f.r, :, :, k, 1)') / 2)
        #= mul!(view(grad_f.grad_omega, :, :, k), view(u, :, :, k), (view(f.r, :, :, k, 1) - view(f.r, :, :, k, 1)')/2) =#
    end =#
    return grad_f.grad_omega
end

function retract_kernel(u_buffer, normsq, n_e)
    k = blockIdx().x
    q = threadIdx().x

    for i = 1:n_e
        normsq[q, k] += abs2(u_buffer[i, q, k])
    end

    for p = 1:n_e
        norm_p = normsq[p, k]
        u_buffer[q, p, k] = u_buffer[q, p, k] / sqrt(norm_p)
        if q > p
            r = 0
            for i = 1:n_e
                r += u_buffer[i, p, k]' * u_buffer[i, q, k]
            end
            for i = 1:n_e
                u_buffer[i, q, k] -= r * u_buffer[i, p, k]
            end
            normsq[q, k] -= abs2(r)
        end
        sync_threads()
    end

end

function retract_kernel(u_buffer, u_buffer_copy, normsq, n_e)
    k = blockIdx().x
    q, i = divrem(threadIdx().x - 1, n_e)
    q += 1
    i += 1

    u_tmp = CuDynamicSharedArray(ComplexGPU, 2n_e * n_e)
    u_tmp[i+(q-1)*n_e] = u_buffer[i, q, k]
    sync_threads()
    u_tmp[i+(q-1)*n_e+n_e^2] = abs2(u_tmp[i+(q-1)*n_e])
    sync_threads()
    #= if i == 1
        for j = 1:n_e
            #= normsq[q, k] += abs2(u_buffer[j, q, k]) =#
            normsq[q, k] += u_tmp[j + (q-1) * n_e + n_e^2]
        end
    end =#
    s = 1
    while s < n_e
        idx = (i - 1) * 2s + 1
        if idx + s <= n_e
            u_tmp[idx+(q-1)*n_e+n_e^2] += u_tmp[idx+s+(q-1)*n_e+n_e^2]
        end
        # end
        sync_threads()
        s *= 2
    end
    normsq[q, k] = u_tmp[1+(q-1)*n_e+n_e^2]
    #= for i = 1:n_e
        normsq[q, k] += abs2(u_buffer[i, q, k])
    end =#
    sync_threads()

    for p = 1:n_e
        norm_p = normsq[p, k]
        u_buffer[q, p, k] = u_buffer[q, p, k] / sqrt(norm_p)
        #= u_tmp[q+(p-1)*n_e, k] = u_tmp[q+(p-1)*n_e] / sqrt(norm_p) =#
        sync_threads()
        #= u_buffer_copy[i, q, k] = u_tmp[i+(p-1)*n_e]' * u_tmp[i+(q-1)*n_e] =#
        u_buffer_copy[i, q, k] = u_buffer[i, p, k]' * u_buffer[i, q, k]
        #= u_tmp[i+(q-1)*n_e+n_e^2] = u_tmp[i+(p-1)*n_e]' * u_tmp[i+(q-1)*n_e] =#
        sync_threads()
        s = 1
        while s < n_e
            # for i in index:stride:div(n_e + 1, 2)
            idx = (i - 1) * 2s + 1
            if idx + s <= n_e
                u_buffer_copy[idx, q, k] += u_buffer_copy[idx+s, q, k]
                #= u_tmp[idx + (q-1)*n_e + n_e^2] += u_tmp[idx+s + (q-1)*n_e + n_e^2] =#
            end
            # end
            sync_threads()
            s *= 2
        end
        r = u_buffer_copy[1, q, k]
        #= r = u_tmp[1 + n_e^2] =#
        if q <= p
            r = 0
        end
        sync_threads()

        u_buffer[i, q, k] -= r * u_buffer[i, p, k]
        #= u_tmp[i+(q-1)*n_e] -= r * u_tmp[i+(p-1)*n_e] =#
        if i == 1
            normsq[q, k] -= abs2(r)
        end
        sync_threads()
    end
    sync_threads()
    #= u_buffer[i, q, k] = u_tmp[i+(q-1)*n_e]
    return nothing =#
end


function retract_gpu!(u_buffer, u, d_u, t, ::QRRetraction)
    n_e, _, n_k = size(u)
    copy!(u_buffer, u)
    axpy!(t, view(d_u, :, :, :), view(u_buffer, :, :, :))
    u_buffer_copy = CUDA.zeros(ComplexGPU, n_e, n_e, n_k)
    #= normsq = CUDA.zeros(Float32, n_e, n_k)
    @cuda threads=n_e blocks=n_k retract_kernel(u_buffer, normsq, n_e)
    nothing =#
    CUDA.@sync begin
        normsq = CUDA.zeros(Float32, n_e, n_k)
        @cuda threads = n_e^2 blocks = n_k shmem = sizeof(ComplexGPU) * 2 * n_e^2 retract_kernel(u_buffer, u_buffer_copy, normsq, n_e)
    end
    nothing
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

