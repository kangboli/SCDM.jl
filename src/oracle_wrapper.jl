export omega_oracle!

const lib_path = joinpath(pathof(SCDM)[1:end-11], "lib", "libwannieroracles")

function omega_oracle!(S::Array{ComplexF64, 4}, U::Array{ComplexF64, 3}, w_list::Vector{Float64},  kplusb::Matrix{Int64}, Nk::Int64, Nb::Int64, Ne::Int64, omega::Vector{Float64}, grad_omega::Array{ComplexF64, 3})

    @ccall lib_path.__oracles_MOD_omega_oracle(S::Ptr{ComplexF64}, U::Ptr{ComplexF64}, w_list::Ptr{Float64}, kplusb::Ptr{Int64}, Nk::Ref{Int64}, Nb::Ref{Int64}, Ne::Ref{Int64}, omega::Ref{Float64}, grad_omega::Ptr{ComplexF64})::Nothing
    #= @ccall "./lib/libwannieroracles".__oracles_MOD_omega_oracle(S::Ptr{ComplexF64}, U::Ptr{ComplexF64}, w_list::Ptr{Float64}, kplusb::Ptr{Int32}, Nk::Ref{Int32}, Nb::Ref{Int32}, Ne::Ref{Int32}, omega::Ref{Float64}, grad_omega::Ptr{ComplexF64})::Nothing =#
    return omega
end


