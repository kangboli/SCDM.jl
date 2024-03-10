export generate_fortran_data, load_problem, write_fortran_files

using FortranFiles

function generate_fortran_data(save_dir::String, target_dir::String)
    MTensor, UTensor, w_list, kplusb, Nk, Nb, N = load_problem(save_dir)
    write_fortran_files(target_dir, MTensor, UTensor, w_list, kplusb, Nk, Nb, N) 
end

function load_problem(save_dir::String)
    wave_functions_list = wave_functions_from_directory(joinpath(save_dir, "aiida.save"))
    ũ = orbital_set_from_save(wave_functions_list)
    N = n_band(ũ)
    _, brillouin_zone = i_kpoint_map(wave_functions_list)
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

    u = ifft(ũ)
    U, _ = scdm_condense_phase(u, collect(1:N))
    M_scdm = gauge_transform(M, U)
    println("$(N) bands")
    sum(i->spread(M_scdm, scheme, i, TruncatedConvolution),1:N) |> println
    UTensor = zeros(ComplexF64, N, N, Nk)
    for k in brillouin_zone
        UTensor[:, :, linear_index(k)] = U[k]
    end
    return MTensor, UTensor, w_list, kplusb, Nk, Nb, N
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
