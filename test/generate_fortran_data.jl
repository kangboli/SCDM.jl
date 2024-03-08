using WTP
using SCDM
using Test
using LinearAlgebra

const test_5_dir = "./test/test_data/test_5"
wave_functions_list = wave_functions_from_directory(joinpath(test_5_dir, "si.save"))
ũ = orbital_set_from_save(wave_functions_list)
N = n_band(ũ)
k_map, brillouin_zone = i_kpoint_map(wave_functions_list)
scheme = CosScheme3D(ũ)

w_list = Vector{Float64}()
shell_list = []
for (w, shell) in zip(weights(scheme), shells(scheme))
    for b in shell
        push!(w_list, w)
        push!(shell_list, b)
    end
end
Nk = length(brillouin_zone)
Nb = length(shell_list)
kplusb = zeros(Int64, Nk, Nb)
for k in brillouin_zone
    for (i, b) in enumerate(shell_list)
        kplusb[linear_index(k), i] = linear_index(k + b)
    end
end

M = neighbor_basis_integral(scheme)
MTensor = zeros(ComplexF64, N, N,  Nk, Nb)
for k in brillouin_zone
    for (i, b) in enumerate(shell_list)
        MTensor[:, :, linear_index(k), i] = M[k, k+b]
    end
end

#= u = ifft(ũ) =#
#= U, columns = scdm_condense_phase(u, collect(1:4)) =#
#= M_scdm = gauge_transform(M, U)
spread(M_scdm, scheme, 1, TruncatedConvolution) =#
UTensor = zeros(ComplexF64, N, N, Nk)
for k in brillouin_zone
    #= UTensor[:, :, linear_index(k)] = U[k] =#
    UTensor[:, :, linear_index(k)] = diagm(ones(4))
end

using FortranFiles

path = "/home/kl935/projects/fast_wannier"
f = FortranFile("$(path)/si_data/w_list.fdat", "w")
write(f, w_list)
close(f)

f = FortranFile("$(path)/si_data/kplusb.fdat", "w")
write(f, kplusb)
close(f)

f = FortranFile("$(path)/si_data/mmn.fdat", "w")
write(f, MTensor)
close(f)

f = FortranFile("$(path)/si_data/amn.fdat", "w")
write(f, UTensor)
close(f)

f = FortranFile("$(path)/si_data/dimensions.fdat", "w")
write(f, Int64(Nk), Int64(Nb), Int64(N))
close(f)

omega = Vector{Float64}([1.0])
grad_omega = zeros(ComplexF64, N, N, Nk)

omega_oracle!(MTensor, UTensor, w_list, kplusb, Nk, Nb, N, omega, grad_omega)
#= ccall((:__oracles_MOD_add_array, "libwannieroracles"), Float64, (Ref{Float64}, Ref{Float64}), 3.0, 4.0) =#

