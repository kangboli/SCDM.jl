export generate_fortran_data, load_problem, write_fortran_files

using FortranFiles

function generate_fortran_data(save_dir::String, target_dir::String)
    m, u, w_list, kplusb, Nk, Nb, N = load_problem(save_dir)
    write_fortran_files(target_dir, m, u, w_list, kplusb, Nk, Nb, N) 
end

function load_problem(save_dir::String)
    wave_functions_list = wave_functions_from_directory(joinpath(save_dir, "aiida.save"))
    orbital_set = orbital_set_from_save(wave_functions_list)
    n_e = n_band(orbital_set)
    _, brillouin_zone = i_kpoint_map(wave_functions_list)
    scheme = CosScheme3D(orbital_set)
    w_list = Vector{Float64}()
    shell_list = []
    for (w, shell) in zip(weights(scheme), shells(scheme))
        for b in shell
            push!(w_list, w)
            push!(shell_list, b)
        end
    end

    n_k = length(brillouin_zone)
    n_b = length(shell_list)
    k_plus_b = zeros(Int64, n_k, n_b)
    for k in brillouin_zone
        for (i, b) in enumerate(shell_list)
            k_plus_b[linear_index(k), i] = linear_index(k + b)
        end
    end

    neighbor_integral = neighbor_basis_integral(scheme) 
    s = zeros(ComplexF64, n_e, n_e,  n_k, n_b)
    for k in brillouin_zone
        for (i, b) in enumerate(shell_list)
            s[:, :, linear_index(k), i] = neighbor_integral[k, k+b]
        end
    end

    orbital_set_real = ifft(orbital_set)
    scdm_gauge, _ = scdm_condense_phase(orbital_set_real, collect(1:n_e))
    M_scdm = gauge_transform(neighbor_integral, scdm_gauge)
    println("$(n_e) bands")
    sum(i->spread(M_scdm, scheme, i, TruncatedConvolution),1:n_e) |> println
    u_scdm = zeros(ComplexF64, n_e, n_e, n_k)
    for k in brillouin_zone
        u_scdm[:, :, linear_index(k)] = scdm_gauge[k]
    end
    return s, u_scdm, w_list, k_plus_b, n_k, n_b, n_e
end


function write_fortran_files(target_dir::String, MTensor::Array{ComplexF64, 4}, UTensor::Array{ComplexF64, 3}, w_list::Vector{Float64}, kplusb::Matrix{Int64}, Nk::Int64, Nb::Int64, N::Int64)
    f = FortranFile("$(target_dir)/w_list.fdat", "w")
    write(f, w_list)
    close(f)

    f = FortranFile("$(target_dir)/kplusb.fdat", "w")
    write(f, kplusb)
    close(f)

    f = FortranFile("$(target_dir)/mmn.fdat", "w")
    write(f, MTensor)
    close(f)

    f = FortranFile("$(target_dir)/amn.fdat", "w")
    write(f, UTensor)
    close(f)

    f = FortranFile("$(target_dir)/dimensions.fdat", "w")
    write(f, Nk, Nb, N)
    close(f)
end
