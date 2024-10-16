export cg
using Printf

function quadratic_fit_1d(f::Function)
    a, negative_2ab, _ = [0.5 -1 0.5; -1.5 2 -0.5; 1.0 0.0 0.0] * f.(0:2)
    b = -negative_2ab / (2a)
    return b, f(b)
end

function make_step!(u_buffer, u_tensor, p_curr, lambda, retract!)
    lambda == 0 && return u_tensor
    retract!(u_buffer, u_tensor, p_curr, float(-lambda))
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
    timers::Dict{Symbol,Int}
    evals::Dict{Symbol,Int}
end

Logger() = Logger(Dict{Symbol,Int}(), Dict{Symbol,Int}())

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

function cg(u, f_no_wrap!, grad_f_no_wrap!, retract_no_wrap!; logger=Logger())
    n_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    n_e, _, n_k = size(u)
    f = record(logger, f_no_wrap!, :f)
    grad_f = record(logger, grad_f_no_wrap!, :grad_f)
    retract! = record(logger, retract_no_wrap!, :retract)
    copy! = record(logger, Base.copy!, :copy)
    #= axpy! = record(logger, LinearAlgebra.BLAS.axpy!, :axpy) =#
    norm = record(logger, LinearAlgebra.norm, :norm)
    f_prev, grad_prev = f(u), grad_f(u)
    res_prev = norm(grad_prev)^2
    p_prev = similar(u)
    fill!(p_prev, 0)
    u_buffer = similar(u)
    fill!(u_buffer, 0)
    iter = 0
    lambda_prev = 1
    #= println(res_prev) =#
    f_curr = f(u)
    start = time_ns()

    while res_prev / (n_k * n_e^2) > 1e-9
        grad_curr = grad_f(u)
        res_curr = norm(grad_curr)^2
        # The fletcher reeves 
        beta = rem(iter, n_e^2) == 0 ? 0 : res_curr / res_prev
        p_curr = grad_curr
        axpy!(beta, p_prev, p_curr)

        function try_scale(scale)
            scale == 0 && return f_curr
            return f(make_step!(u_buffer, u, p_curr, scale * lambda_prev, retract!))
        end
        scale, f_tmp = quadratic_fit_1d(try_scale)
        lambda_curr = scale * lambda_prev
        if lambda_curr > 0 && f_tmp < f_prev
            f_curr = f_tmp
        else # fall back to a gradient descent if the function is not locally convex.
            axpy!(-beta, p_prev, p_curr)
            f_curr, lambda_curr, _ = line_search!(u_buffer, u, f, f_curr, p_curr, res_curr, 2lambda_prev, retract!)
            axpy!(beta, p_prev, p_curr)
        end

        copy!(u, u_buffer)

        lambda_prev = lambda_curr
        f_prev = f_curr
        copy!(p_prev, p_curr)
        res_prev = res_curr
        if mod(iter, 10) == 0
            @printf "iter: %d, f_curr: %.3f, lambda: %.3e, res: %.3e\n" iter f_curr lambda_curr res_curr
        end
        iter += 1
    end
    finish = time_ns()
    BLAS.set_num_threads(n_threads)
    println("value:   ", f_curr)
    println("iter:    ", iter)
    println("f_eval:  ", logger.evals[:f])
    println("f:       ", logger.timers[:f] / 1e9)
    println("grad_f:  ", logger.timers[:grad_f] / 1e9)
    println("retract: ", logger.timers[:retract] / 1e9)
    #= println("axpy:    ", logger.timers[:axpy] / 1e9) =#
    println("copy:    ", logger.timers[:copy] / 1e9)
    println("norm:    ", logger.timers[:norm] / 1e9)
    println("n_norm:  ", logger.evals[:norm])
    println("compute: ", total_time(logger) / 1e9)
    println("wall:    ", (finish - start) / 1e9)
    return u
end
#
