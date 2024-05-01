export ILAOptimizer, FletcherReeves, logger, AcceleratedGradientDescent

"""
A three-points fitting method for conjugate gradient.

f(λ) ≈ a (λ - b)² + c around λ = 1
f(λ) ≈ a λ² - 2ab λ + b² + c

| 0  0  1 | | a      | = | f(0) |
| 1  1  1 | | -2ab   |   | f(1) |
| 4  2  1 | | b² + c |   | f(2) |

```julia
quadratic_fit_1d(x->2(x-3)^2 + 3)

# output

(3.0, 3.0)
```
"""
function quadratic_fit_1d(f::Function)
    a, negative_2ab, _ = [.5 -1 .5; -1.5 2 -.5; 1.0 .0 .0] * f.(0:2)
    b = -negative_2ab / (2a)
    return b, f(b)
end

function all_spread(U, scheme, ::Type{Branch}) where {Branch}
    M = gauge_transform(neighbor_basis_integral(scheme), U)
    N = n_band(M)
    return map(1:N) do n
        spread(M, scheme, n, Branch)
    end
end

function all_center(U, scheme, ::Type{Branch}) where {Branch}
    M = gauge_transform(neighbor_basis_integral(scheme), U)
    N = n_band(M)
    return map(1:N) do n
        center(M, scheme, n, Branch)
    end
end

function all_spread(U, scheme, ::Type{TruncatedConvolution})
    M = gauge_transform(neighbor_basis_integral(scheme), U)
    N = n_band(M)
    brillouin_zone = collect(grid(scheme))
    ρ̃(b) = sum(k -> diag(M[k, k+b]), brillouin_zone) / length(brillouin_zone)

    sum(zip(weights(scheme), shells(scheme))) do (w, shell)
        sum(b -> 2w * (ones(N) - abs.(ρ̃(b))), shell)
    end
end

"""
    ILAOptimizer(scheme)

An amateur manifold optimizer with gradient descent, accelerated gradient
descent, and conjugate gradient. The project aims to integrate with `manopt.jl`
and deprecate the custom optimizer except for debugging.

Example:

```jldoctest orbital_set
julia> optimizer = ILAOptimizer(scheme);
julia> U_optimal = optimizer(U, TruncatedConvolution, FletcherReeves);
julia> M_optimal = gauge_transform(neighbor_basis_integral(scheme), U_optimal);
julia> sum(i->spread(M_optimal, scheme, i, TruncatedConvolution), 1:4)
24.206845069491276
```

Optimization with `W90BranchCut` is not included because it's too much work for 
`Documenter.jl`.
"""
struct ILAOptimizer
    scheme::CosScheme
    meta::Dict{Symbol, Any}
    logger::Dict{Symbol,Vector}
end

ILAOptimizer(scheme::CosScheme) = ILAOptimizer(scheme,
    Dict{Symbol, Any}(),
    Dict{Symbol, Vector}(
        :w90_branch_cut_center => Vector{Matrix{Float64}}(),
        :truncated_convolution_center => Vector{Matrix{Float64}}(),
        :convolutional_center => Vector{Matrix{Float64}}(),
        :w90_branch_cut_spread => Vector{Vector{Float64}}(),
        :truncated_convolution_spread => Vector{Vector{Float64}}(),
        :convolutional_spread => Vector{Vector{Float64}}(),
        :step_size => Vector{Vector{Float64}}(),
        :Ω  => Vector{Float64}(),
    ))

scheme(optimizer::ILAOptimizer) = optimizer.scheme
logger(optimizer::ILAOptimizer) = optimizer.logger

"""
    make_step(U, ΔW, α)

Make a step from `log(U)` in the direction of `ΔW` by size `α`.  Keep in mind
that the change in `U` is an approximate since `α ΔW` does not commute with
`log(U)`, and the exponentiation cannot exactly be separated.
"""
function make_step(U::Gauge, ΔW, α)
    brillouin_zone = grid(U)
    new_elements = elements(map(k -> U[k] * cis(-Hermitian(1im * α * ΔW[k])), brillouin_zone))
    V = Gauge{typeof(brillouin_zone)}(brillouin_zone, new_elements)
    return V
end

"""
    line_search(U, f, ∇f, ∇f², α, α_0 = 2)

Perform a line search in the direction of the gradient.
"""
function line_search(U, f, ∇f, ∇f², α; α_0 = 2)
    Ω, ∇Ω = f(U), ∇f(U)
    ∇Ω² = ∇f²(∇Ω)

    while true
        Q = Ω - 0.5α * ∇Ω²
        V = make_step(U, ∇Ω, α)
        Ω_V = f(V)

        # This is disabled to test that tdc never needs this..
        # discontinuous = ∇Ω² > 1e-7 && α < 1e-3
        # discontinuous && return let α = α_0, V = make_step(U, ∇Ω, α), Ω_V = f(V)
        #     print("✽")
        #     V, Ω_V, α, ∇Ω²
        # end

        Ω_V > Q || return V, Ω_V, α, ∇Ω²
        α /= 2
    end
end

"""
The (tampered) Fletcher Reeves conjugate gradient algorithm.

For an explanation of this conjugate gradient algorithm,
see: http://www.mymathlib.com/optimization/nonlinear/unconstrained/fletcher_reeves.html

The algorithm is adapted slightly since our objective function is not convex.
We added simple line searches when the quadratic fit turns out concave.
"""
struct FletcherReeves end

"""
The accelerated gradient descent algorithm. The acceleration starts when the
gradient becomes small enough.
"""
struct AcceleratedGradientDescent end

function (optimizer::ILAOptimizer)(U::Gauge, ::Type{Branch}, ::Type{AcceleratedGradientDescent}; n_iteration = Inf, α_0 = 2, ϵ = 1e-7, logging = false) where {Branch}
    N = optimizer |> scheme |> neighbor_basis_integral |> n_band
    brillouin_zone = grid(U)
    f = U -> sum(all_spread(U, scheme(optimizer), Branch))
    ∇f = U -> let M = gauge_transform(neighbor_basis_integral(scheme(optimizer)), U)
        Gauge{typeof(brillouin_zone)}(brillouin_zone, elements(gauge_gradient(M, scheme(optimizer), brillouin_zone, Branch)))
    end
    ∇f² = ∇f -> sum(k -> norm(reshape(∇f[k], N^2))^2, brillouin_zone)
    α = α_0
    t = 1
    current_iteration = -1
    while n_iteration > current_iteration
        current_iteration += 1
        X, Ω, α, ∇Ω² = line_search(U, f, ∇f, ∇f², α, α_0 = α_0)
        ∇Ω² / (length(brillouin_zone) * N^2) < ϵ && break
        logging && log(optimizer, U, current_iteration, Ω, ∇Ω², α)
        if ∇Ω² / (length(brillouin_zone) * N^2) > 1e-3
            α = 2α
            U = X
            continue
        end
        t_next = (1 + sqrt(1 + 4t^2)) / 2
        U = X + (t - 1) / t_next * (X - U)
        t = t_next
    end
    return U
end


function (optimizer::ILAOptimizer)(U::Gauge, ::Type{Branch}, ::Type{FletcherReeves}; ϵ = 1e-7, logging = false) where {Branch}
    N = optimizer |> scheme |> neighbor_basis_integral |> n_band
    brillouin_zone = grid(U)
    f = U -> sum(all_spread(U, scheme(optimizer), Branch))
    ∇f = U -> let M = gauge_transform(neighbor_basis_integral(scheme(optimizer)), U)
        Gauge{typeof(brillouin_zone)}(brillouin_zone, elements(gauge_gradient(M, scheme(optimizer), brillouin_zone, Branch)))
    end
    ∇f² = ∇f -> sum(k -> norm(reshape(∇f[k], N^2))^2, brillouin_zone)
    current_iteration = 0
    h_old, g_old = f(U), ∇f(U)
    g²_old = ∇f²(g_old)
    v_old = Gauge(brillouin_zone, N)
    λ_old = 2

    while true
        g²_old / (length(brillouin_zone) * N^2) < ϵ && break
        g = ∇f(U)
        g² = ∇f²(g)
        α = rem(current_iteration, N^2) == 0 ? 0 : g² / g²_old
        v = Gauge{typeof(brillouin_zone)}(brillouin_zone, elements(map(k -> g[k] + α * v_old[k], brillouin_zone)))
        λ, h = quadratic_fit_1d(λ -> make_step(U, v, λ) |> f)
        if λ > 0 && h < h_old
            U = make_step(U, v, λ)
            h = f(U)
        else
            U, h, λ, _ = line_search(U, f, ∇f, ∇f², 2λ_old)
        end

        logging && log(optimizer, U, current_iteration, h, g², λ)
        λ_old, h_old, v_old, g_old, g²_old = λ, h, v, g, g²
        current_iteration += 1
    end
    optimizer.meta[:n_iteration] = current_iteration
    return U
end

function log(optimizer, U, current_iteration, Ω, ∇Ω², α)
    println("Iteration: $(current_iteration)")
    println("Ω: $(Ω)")
    println("∇Ω²: $(∇Ω²)")
    println("α: $(α)")
    append!(logger(optimizer)[:Ω], [Ω])
    # println()

    # N = optimizer |> scheme |> neighbor_basis_integral |> n_band
    append!(logger(optimizer)[:truncated_convolution_spread], [all_spread(U, scheme(optimizer), TruncatedConvolution)])
    append!(logger(optimizer)[:w90_branch_cut_spread], [all_spread(U, scheme(optimizer), W90BranchCut)])
    # append!(logger(optimizer)[:truncated_convolution_center], [hcat(all_center(U, scheme(optimizer), TruncatedConvolution)...)])
    # append!(logger(optimizer)[:w90_branch_cut_center], [hcat(all_center(U, scheme(optimizer), W90BranchCut)...)])


    # # haskey(optimizer.meta, :truncated_convolution_spread)  || (optimizer.meta[:truncated_convolution_spread] = Vector{Vector{Float64}}())
    # # haskey(optimizer.meta, :w90_branch_cut_spread)  || (optimizer.meta[:w90_branch_cut_spread] = Vector{Vector{Float64}}())
    # # haskey(optimizer.meta, :convolutional_spread)  || (optimizer.meta[:convolutional_spread] = Vector{Vector{Float64}}())

    # # haskey(optimizer.meta, :truncated_convolution_center)  || (optimizer.meta[:truncated_convolution_center] = Vector{Matrix{Float64}}())
    # # haskey(optimizer.meta, :w90_branch_cut_center)  || (optimizer.meta[:w90_branch_cut_center] = Vector{Matrix{Float64}}())
    # # haskey(optimizer.meta, :convolutional_center)  || (optimizer.meta[:convolutional_center] = Vector{Matrix{Float64}}())


    # mod(current_iteration, 10) == 0 || return
    haskey(optimizer.meta, :ũ) || return
    ũ = set_gauge(optimizer.meta[:ũ], U)
    r̃2 = optimizer.meta[:r̃2]
    ρ̃ = reciprocal_densities(ũ)

    convolutional_center = zeros(3, length(ρ̃))
    convolutional_spread = zeros(length(ρ̃))
    for i in 1:length(ρ̃)
        c, σ² = center_spread(ρ̃[i], r̃2)
        convolutional_center[:, i] = c
        convolutional_spread[i] = σ²
    end
    append!(logger(optimizer)[:convolutional_center], [convolutional_center])
    append!(logger(optimizer)[:convolutional_spread], [convolutional_spread])
end
