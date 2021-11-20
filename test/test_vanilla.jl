using WTP
using SCDM

const test_1_dir = "./test/test_data/test_1"

wave_functions_list = wave_functions_from_directory(joinpath(test_1_dir, "si.save"))
wannier = wannier_from_save(wave_functions_list)

# wannier = wannier_from_unk_dir(joinpath(test_1_dir, "unk"), wave_functions_list)
k_map, brillouin_zone = i_kpoint_map(wave_functions_list)
wannier_real = ifft(wannier)

U = scdm_condense_phase(wannier_real, collect(1:20));

U[brillouin_zone[0,0,0]]

amn = AMN(joinpath(test_1_dir, "unk", "si.amn"))

U_matlab = Gauge(brillouin_zone, amn, k_map, false)

for k in brillouin_zone
    @test norm(U[k] - U_matlab[k]) < 1e-6
end
