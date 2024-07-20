using SCDM, LinearAlgebra, Random, Profile, WTP

src_dir = "./test/scdm_dataset/N2Na2O6"

wave_functions_list = wave_functions_from_directory(joinpath(src_dir, "aiida.save"))
k_map, brillouin_zone = i_kpoint_map(wave_functions_list)
mmn = MMN(joinpath(src_dir, "aiida.mmn"))
ki = NeighborIntegral(mmn, k_map)
f, orbital_set = load_problem(joinpath(src_dir, "aiida.save"))
scheme = CosScheme3D(orbital_set; integrate=false)
for k in brillouin_zone
    for (i, b) in enumerate(vcat(shells(scheme)...))
        f.s[:, :, linear_index(k), i] = ki[k, k + b]
    end
end

#= f = make_f(s, w_list, k_plus_b, n_k, n_b, n_e, n_e) =#
grad_f = make_grad_f(f)

#= Random.seed!(16)
u = zeros(ComplexF64, n_e, n_e, n_k)

for k in 1:n_k
    a = rand(n_e, n_e)
    u[:, :, k] = let (u, _, v) = svd(a)
        u * v'
    end
end =#

@time cg(random_gauge(f), f, grad_f, qr_retract!);


#= orbital_set = orbital_set_from_save(wave_functions_list, domain_scaling_factor=1)
n_e = n_band(orbital_set)
scheme = CosScheme3D(orbital_set)

scheme.neighbor_basis_integral[brillouin_zone[0, 0, 0], brillouin_zone[0, 0,1]][1, 3]

s = zeros(ComplexF64, n_e, n_e, n_k, n_b)

k_plus_b = zeros(Int64, n_k, n_b)
k_minus_b = zeros(Int64, n_k, n_b)
for k in brillouin_zone
    for (i, b) in enumerate(shell_list)
        k_plus_b[linear_index(k), i] = linear_index(k + b)
        k_minus_b[linear_index(k), i] = linear_index(k - b)
    end
end
 =#

