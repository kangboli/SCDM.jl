using WTP
using SCDM
using Test
using LinearAlgebra

const test_1_dir = "./test/test_data/test_1"
wave_functions_list = wave_functions_from_directory(joinpath(test_1_dir, "si.save"))
ũ = orbital_set_from_save(wave_functions_list)
k_map, brillouin_zone = i_kpoint_map(wave_functions_list)
scheme = CosScheme3D(ũ)
