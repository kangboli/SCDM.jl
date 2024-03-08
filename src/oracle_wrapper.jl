export omega_oracle!

function omega_oracle!(S::Array{ComplexF64, 4}, U::Array{ComplexF64, 3}, w_list::Vector{Float64},  kplusb::Matrix{Int32}, Nk::Int32, Nb::Int32, Ne::Int32, omega::Vector{Float64}, grad_omega::Array{ComplexF64, 3})
    @ccall "libwannieroracles".__oracles_MOD_omega_oracle(S::Ptr{ComplexF64}, U::Ptr{ComplexF64}, w_list::Ptr{Float64}, kplusb::Ptr{Int32}, Nk::Ref{Int32}, Nb::Ref{Int32}, Ne::Ref{Int32}, omega::Ref{Float64}, grad_omega::Ptr{ComplexF64})::Nothing
    return omega
end


