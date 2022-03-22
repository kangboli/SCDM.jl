using SCDM
using WTP
using Manifolds
using Manopt
using LinearAlgebra


wave_functions_list = wave_functions_from_directory(joinpath(test_5_dir, "si.save"))
ũ = orbital_set_from_save(wave_functions_list)
brillouin_zone = grid(ũ)
k_map, _ = i_kpoint_map(wave_functions_list)

s = CosScheme3D(ũ)

amn = AMN(joinpath(test_5_dir, "output/pw2wan/si.amn"))
U = Gauge(grid(ũ), amn, k_map)

M = gauge_transform(neighbor_basis_integral(s), U)
sum(n -> spread(M, s, n, TruncatedConvolution), 1:4)

ℳ = PowerStiefel(U)
p = PowerStiefelPoint(U)
U_1 = Gauge(p, brillouin_zone)
# check_vector(ℳ, p, gradF(ℳ, p))

exp(ℳ, p, gradF(ℳ, p))

F = make_objective(neighbor_basis_integral(s), s, brillouin_zone, TruncatedConvolution)
gradF = make_gradient(neighbor_basis_integral(s), s, brillouin_zone, TruncatedConvolution)


F(ℳ, exp(ℳ, p, 0.1 * gradF(ℳ, p)))
F(ℳ, p)
F(ℳ, retract(ℳ, p, -0.2 * gradF(ℳ, p), CayleyRetraction()))


# retract(ℳ, p, -0.08 * gradF(ℳ, p))

result = gradient_descent(ℳ, F, gradF, p; 
    stepsize=ConstantStepsize(0.01),
    debug=[:Iteration, " | ", :Cost, "\n", :Stop],
    stopping_criterion=StopWhenGradientNormLess(1e-3),
    retraction_method=CayleyRetraction(),
    # vector_transport_method=ParallelTransport(),
    )

result = conjugate_gradient_descent(ℳ, F, gradF, p; 
    stepsize=ConstantStepsize(0.01),
    debug=[:Iteration, " | ", :Cost, "\n", :Stop],
    stopping_criterion=StopWhenGradientNormLess(1e-3),
    retraction_method=ExponentialRetraction(),
    vector_transport_method=ProjectionTransport(),
    coefficient=FletcherReevesCoefficient()
    )
F(ℳ, result)

# function Manifolds.retract(ℳ::PowerManifold{Manifolds.ManifoldsBase.ComplexNumbers() ,M}, p, X, T::AbstractRetractionMethod) where {M <: Stiefel}
#     retracted = deepcopy(p)
#     for k = 1:size(p, 3)
#         # retracted[:, :, k] = retract(ℳ.manifold, p[:, :, k], X[:, :, k], T)
#         retracted[:, :, k] = p[:, :, k] * cis(-1im * X[:, :, k])
#     end
#     return retracted
# end

# function Base.exp(ℳ::PowerManifold{Manifolds.ManifoldsBase.ComplexNumbers() ,M}, p, X) where {M <: Stiefel}
#     retracted = deepcopy(p)
#     for k = 1:size(p, 3)
#         # retracted[:, :, k] = retract(ℳ.manifold, p[:, :, k], X[:, :, k], T)
#         retracted[:, :, k] = p[:, :, k] * cis(-1im * X[:, :, k])
#     end
#     println("My Exp called")
#     return retracted
# end