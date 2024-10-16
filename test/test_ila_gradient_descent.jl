using WTP
using Test
using LinearAlgebra
using SCDM

@testset "ILA No Guess Optimization" begin
    wave_functions_list = wave_functions_from_directory(joinpath(test_5_dir, "si.save"))
    ũ = orbital_set_from_save(wave_functions_list)
    brillouin_zone = grid(ũ)
    homecell = transform_grid(orbital_grid(ũ))
    ## Get the map of kpoints.

    k_map, _ = i_kpoint_map(wave_functions_list)

    # No guess optimization
    U = Gauge(grid(ũ), n_band(ũ))
    s = CosScheme3D(ũ)
    optimizer = ILAOptimizer(s)
    # supercell = expand(homecell, [size(brillouin_zone)...])
    # r̃2 = fft(map(r->norm(r)^2, supercell) , false)

    # optimizer.meta[:r̃2] = r̃2
    # optimizer.meta[:ũ] = ũ
    # optimizer.meta[:ila_spread] = Vector{Vector{Float64}}()
    # optimizer.meta[:convolutional_spread] = Vector{Vector{Float64}}()
    # optimizer.meta[:center_difference] = Vector{Vector{Float64}}()

    U_optimal = optimizer(U, TruncatedConvolution, AcceleratedGradientDescent; α_0=1, ϵ=1e-7)
    U_optimal = optimizer(U, RTDC, FletcherReeves; ϵ=1e-7, logging=true);
    U_optimal = optimizer(U, TruncatedConvolution, FletcherReeves; ϵ=1e-7, logging=true);
    # optimizer.meta[:ila_spread]
    M_optimal = gauge_transform(neighbor_basis_integral(s),  U_optimal)
    cartesian.((c->reset_overflow(snap(homecell, c))).((i->center(M_optimal, scheme, i)).(1:4)))
    @test all(s->isapprox(s, 6.052, atol=1e-1), (i->spread(M_optimal, scheme, i, TruncatedConvolution)).(1:4))
end

# optimizer.meta[:r̃2] = r̃2
# optimizer.meta[:ũ] = ũ
# optimizer.meta[:ila_spread] = Vector{Vector{Float64}}()
# optimizer.meta[:convolutional_spread] = Vector{Vector{Float64}}()
# optimizer.meta[:center_difference] = Vector{Vector{Float64}}()