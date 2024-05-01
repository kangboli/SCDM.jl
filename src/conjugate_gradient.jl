export cg

function quadratic_fit_1d(f::Function)
    a, negative_2ab, _ = [0.5 -1 0.5; -1.5 2 -0.5; 1.0 0.0 0.0] * f.(0:2)
    b = -negative_2ab / (2a)
    return b, f(b)
end

function make_step!(u_buffer, u_tensor::Array{ComplexF64,3}, p_curr, lambda, retract!)
    lambda == 0 && return u_tensor
    retract!(u_buffer, u_tensor, p_curr, float(-lambda), QRRetraction())
    return u_buffer
end

function line_search!(u_buffer, u_tensor, f, f_curr, grad_curr, res_curr, step, retract!)
    while true
        Q = f_curr - 0.5step * res_curr
        make_step!(u_buffer, u_tensor, grad_curr, step, retract!)
        f_v = f(u_buffer)
        f_v > Q || return f_v, step, res_curr
        step /= 2
    end
end

struct Logger
    timers::Dict{Symbol, Int}
    evals::Dict{Symbol, Int}
end

Logger() = Logger(Dict{Symbol, Int}(), Dict{Symbol, Int}())

total_time(l::Logger) = sum(values(l.timers))

function record(l::Logger, f_no_wrap::Any, name::Symbol)
    l.evals[name] = 0
    l.timers[name] = 0
    function f(u...)
        l.evals[name] += 1
        start = time_ns()
        result = f_no_wrap(u...)
        finish = time_ns()
        l.timers[name] += (finish - start)
        return result
    end
    return f
end

"""

Memory consumption:

The current implementation keeps two copies of the overlap matrices to speed up
the gradient calculation by reusing the intermediate values from the forward
evaluation. Whether this tradeoff is worthwhile depends on how often the gradient
is evaluated compared to the objective function, which depends on the optimizer.
Exploring the tradeoff is currently not a priority.
"""

function cg(u_tensor::Array{ComplexF64,3}, f_no_wrap!, grad_f_no_wrap!, retract!, N::Int, Nk::Int; logger=Logger())
    normsq = x -> norm(x)^2
    f = record(logger, f_no_wrap!, :f)
    grad_f = record(logger, grad_f_no_wrap!, :grad_f)
    retract! = record(logger, retract!, :retract)
    copy! = record(logger, Base.copy!, :copy)
    axpy! = record(logger, LinearAlgebra.BLAS.axpy!, :axpy)
    normsq = record(logger, normsq, :normsq)
    f_prev, grad_prev = f(u_tensor), grad_f(u_tensor)
    res_prev = normsq(grad_prev)
    p_prev = zeros(ComplexF64, size(u_tensor))
    u_buffer = zeros(ComplexF64, size(u_tensor))
    iter = 0
    lambda_prev = 1
    f_curr = f_prev
    start = time_ns()

    while res_prev / (Nk * N^2) > 1e-7
        grad_curr = grad_f(u_tensor)
        res_curr = normsq(grad_curr)
        # The fletcher reeves 
        beta = rem(iter, N^2) == 0 ? 0 : res_curr / res_prev
        p_curr = grad_curr
        axpy!(beta, p_prev, p_curr)

        function try_scale(scale)
            scale == 0 && return f_curr
            return f(make_step!(u_buffer, u_tensor, p_curr, scale * lambda_prev, retract!))
        end
        scale, f_tmp = quadratic_fit_1d(try_scale)
        lambda_curr = scale * lambda_prev
        if lambda_curr > 0 && f_tmp < f_prev
            f_curr = f_tmp
        else # fall back to a gradient descent if the function is not locally convex.
            axpy!(-beta, p_prev, p_curr)
            f_curr, lambda_curr, _ = line_search!(u_buffer, u_tensor, f, f_curr, p_curr, res_curr, 2lambda_prev, retract!)
            axpy!(beta, p_prev, p_curr)
        end

        copy!(u_tensor, u_buffer)

        lambda_prev = lambda_curr
        f_prev = f_curr
        copy!(p_prev, p_curr)
        res_prev = res_curr
        #= println(f_curr) =#
        iter += 1
    end
    finish = time_ns()
    println("total: ", (finish - start) / 1e9)
    println(iter)
    println(logger.evals[:f])
    println("f: ",  logger.timers[:f] / 1e9)
    println("grad_f: ", logger.timers[:grad_f] / 1e9)
    println("retract: ", logger.timers[:retract] / 1e9)
    println("axpy: ", logger.timers[:axpy] / 1e9)
    println("copy: ", logger.timers[:copy] / 1e9)
    println("normsq: ", logger.timers[:normsq] / 1e9)
    println(total_time(logger) / 1e9)
    return u_tensor
end
#
