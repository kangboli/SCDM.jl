export generate_fortran_data, load_problem, write_fortran_files, identity_gauge, scdm_gauge

using FortranFiles

function generate_fortran_data(save_dir::String, target_dir::String)
    m, u, w_list, kplusb, Nk, Nb, N = load_problem(save_dir)
    write_fortran_files(target_dir, m, u, w_list, kplusb, Nk, Nb, N)
end

"""
    load_problem(save_dir, bands=nothing)

This function loads from a `.save` directory and returns a number of tensors.
It loosely correspond to `pw2wannier.x`.

"""
function load_problem(save_dir::String, bands=nothing)
    wave_functions_list = wave_functions_from_directory(save_dir)
    if bands === nothing
        orbital_set = orbital_set_from_save(wave_functions_list, domain_scaling_factor=1)
    else
        orbital_set = orbital_set_from_save(wave_functions_list, bands=bands, domain_scaling_factor=1)
    end
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
    k_minus_b = zeros(Int64, n_k, n_b)
    for k in brillouin_zone
        for (i, b) in enumerate(shell_list)
            #= k_plus_b[linear_index(k), i] = linear_index(k + b)
            k_minus_b[linear_index(k), i] = linear_index(k - b) =#
            k_plus_b[linear_index(k), i] = linear_index(reset_overflow(k + b))
            k_minus_b[linear_index(k), i] = linear_index(reset_overflow(k - b))
        end
    end

    neighbor_integral = neighbor_basis_integral(scheme)
    s = zeros(ComplexF64, n_e, n_e, n_k, n_b)
    for k in brillouin_zone
        for (i, b) in enumerate(shell_list)
            s[:, :, linear_index(k), i] = neighbor_integral[k, k+b]
        end
    end

    #= orbital_set_real = ifft(orbital_set)
    scdm_gauge, _ = scdm_condense_phase(orbital_set_real, collect(1:n_e))
    M_scdm = gauge_transform(neighbor_integral, scdm_gauge)
    println("$(n_e) bands")
    sum(i -> spread(M_scdm, scheme, i, TruncatedConvolution), 1:n_e) |> println =#
    #= return s, u_init, w_list, k_plus_b, k_minus_b, n_k, n_b, n_e, hcat(coordinates.(shell_list)...) =#
    shell_list = hcat(coordinates.(shell_list)...)
    return make_f(s, w_list, shell_list, k_plus_b, k_minus_b, n_k, n_b, n_e, n_e), orbital_set
end

function scdm_gauge(orbital_set)
    n_e = n_band(orbital_set)
    brillouin_zone = grid(orbital_set)
    n_k = length(brillouin_zone)
    orbital_set_real = ifft(orbital_set)
    scdm_gauge, _ = scdm_condense_phase(orbital_set_real, collect(1:n_e))
    u_scdm = zeros(ComplexF64, n_e, n_e, n_k)
    for k in brillouin_zone
        u_scdm[:, :, linear_index(k)] = scdm_gauge[k]
    end
    return u_scdm
end

function identity_gauge(f::OracleF)
    u_init = zeros(ComplexF64, f.n_e, f.n_e, f.n_k)
    for k in 1:f.n_k
        u_init[:, :, k] = Diagonal(ones(ComplexF64, f.n_e))
    end
    return u_init
end


function write_fortran_files(target_dir::String, MTensor::Array{ComplexF64,4}, UTensor::Array{ComplexF64,3}, w_list::Vector{Float64}, kplusb::Matrix{Int64}, Nk::Int64, Nb::Int64, N::Int64)
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
