
using Manopt
using Manifolds

export PowerStiefel, PowerStiefelPoint, make_objective, make_gradient

"""
Incooperate Manopt.jl for iterative optimization.
"""

"""
    PowerStiefel(U)

Construct a Power Stiefel manifold from a gauge ``U``.
"""
PowerStiefel(U::Gauge) = PowerManifold(Stiefel(size(elements(U)[1])..., Manifolds.ManifoldsBase.ComplexNumbers()), length(grid(U)))

"""
    PowerStiefelPoint(U)

Construct a point on the corresponding Power Stiefel manifold from a gauge ``U``.
"""
PowerStiefelPoint(U::Gauge) = cat([U[k] for k in grid(U)]..., dims=3)

"""
    Gauge(p, brillouin_zone)

Construct a gauge from a point on the Stiefel manifold and a brillouin_zone.
"""
function WTP.Gauge(p::AbstractArray, brillouin_zone::BrillouinZone) 
    U = Gauge(brillouin_zone)
    for i = 1:size(p, 3)
        U[brillouin_zone(i)] = p[:, :, i]
    end
    return U
end

"""
    make_objective(M, scheme, brillouin_zone, Formula)

Make the function that approximates the spread for optimization. 
"""
function make_objective(M::NeighborIntegral, scheme::ApproximationScheme, brillouin_zone::BrillouinZone, ::Type{Formula}) where Formula
    function F(ℳ, p)
        U = Gauge(p, brillouin_zone)
        M_new = gauge_transform(M, U)
        sum(n->spread(M_new, scheme, n, Formula), 1:n_band(M))
    end
    return F
end

"""
    make_gradient(M, scheme, brillouin_zone, Formula)

Make the function that approximates the spread gradient for optimization. 
"""
function make_gradient(M::NeighborIntegral, scheme::ApproximationScheme, brillouin_zone::B, ::Type{Formula}) where {Formula, B <: BrillouinZone}
    function gradF(ℳ, p)    
        U = Gauge(p, brillouin_zone)
        M_new = gauge_transform(M, U)
        G = -gauge_gradient(M_new, scheme, brillouin_zone, Formula) |> Gauge
        return G |> PowerStiefelPoint
    end
end

"""
Exponential retraction (Full Pade approximation).
"""
function Manifolds.retract!(::PowerManifold{ℂ, M}, q, p, X, ::ExponentialRetraction) where M<:(Stiefel{n, k, ℂ} where {n, k})
    for k = 1:size(p, 3)
        q[:, :, k] = p[:, :, k] * cis(-Hermitian(1im * X[:, :, k]))
        # q[:, :, k] = p[:, :, k] * exp(X[:, :, k])
    end
    return q
end


"""
Cayley retraction (first order Pade approximation)
"""
function Manifolds.retract!(::PowerManifold{ℂ, M}, q, p, X, ::CayleyRetraction) where M<:(Stiefel{n, k, ℂ} where {n, k})
    for k = 1:size(p, 3)
        W = X[:, :, k]
        q[:, :, k] = p[:, :, k] * ((I - W) \ (I + W))

        # P = I # - 1/2 * p[:, :, k] * p[:, :, k]' # May not be necessary
        # W = X[:, :, k] * p[:, :, k]' - p[:, :, k] * X[:, :, k]' 
        # q[:, :, k] = p[:, :, k] * ((I - X[:, :, k]) \ (I + X[:, :, k]))
    end
    return q
end


function Manifolds.norm(::PowerManifold{ℂ, M}, p, X) where M<:(Stiefel{n, k, ℂ} where {n, k})
    let v = reshape(X, prod(size(X)))
        norm(v)
    end
end