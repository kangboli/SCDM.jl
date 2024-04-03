using WTP
using SCDM
using Test
using LinearAlgebra
using FortranFiles

si_src_dir = "./test/scdm_dataset/Si"
si_tgt_dir = "/home/kl/projects/fast_wannier/data/Si"
generate_fortran_data(si_src_dir, si_tgt_dir)

gaas_src_dir = "./test/scdm_dataset/GaAs"
gaas_tgt_dir = "/home/kl/projects/fast_wannier/data/GaAs"
generate_fortran_data(gaas_src_dir, gaas_tgt_dir)

batio3_src_dir = "./test/scdm_dataset/BaTiO3"
batio3_tgt_dir = "/home/kl/projects/fast_wannier/data/BaTiO3"
generate_fortran_data(batio3_src_dir, batio3_tgt_dir)

fli_src_dir = "./test/scdm_dataset/FLi"
fli_tgt_dir = "/home/kl/projects/fast_wannier/data/FLi"
generate_fortran_data(fli_src_dir, fli_tgt_dir)

brna_src_dir = "./test/scdm_dataset/BrNa"
brna_tgt_dir = "/home/kl/projects/fast_wannier/data/BrNa"
generate_fortran_data(brna_src_dir, brna_tgt_dir)

he_src_dir = "./test/scdm_dataset/He"
he_tgt_dir = "/home/kl/projects/fast_wannier/data/He"
generate_fortran_data(he_src_dir, he_tgt_dir)

sih4_src_dir = "./test/scdm_dataset/SiH4"
sih4_tgt_dir = "/home/kl/projects/fast_wannier/data/SiH4"
generate_fortran_data(sih4_src_dir, sih4_tgt_dir)

#= ccall((:__oracles_MOD_add_array, "libwannieroracles"), Float64, (Ref{Float64}, Ref{Float64}), 3.0, 4.0) =#

