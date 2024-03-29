using WTP
using SCDM
using Test
using LinearAlgebra

const test_1_dir = "./test/test_data/test_1"
wave_functions_list = wave_functions_from_directory(joinpath(test_1_dir, "si.save"))
ũ = orbital_set_from_save(wave_functions_list)

# wannier = wannier_from_unk_dir(joinpath(test_1_dir, "unk"), wave_functions_list)
k_map, brillouin_zone = i_kpoint_map(wave_functions_list)
u = ifft(ũ)
homecell = orbital_grid(u)
columns = quadratic_pivot_scdm(u, collect(1:4), 2)
[homecell(c) for c in columns]

x_max(homecell) - x_min(homecell)
Ψ_Γ = u[brillouin_zone[0, 0, 0]]
length(grid(first(Ψ_Γ)))

U, columns = scdm_condense_phase(u, collect(1:4))

amn = AMN(joinpath(test_1_dir, "unk", "si.amn"))

U_matlab = Gauge(brillouin_zone, amn, k_map, false)

for k in brillouin_zone
    @test norm(U[k] - U_matlab[k]) < 1e-6
end
