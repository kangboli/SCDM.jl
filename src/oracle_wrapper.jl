export f_oracle!, grad_f_oracle!, project!, retract!

const lib_path = joinpath(pathof(SCDM)[1:end-11], "lib", "libwannieroracles")

#= function omega_oracle!(S::Array{ComplexF64,4}, R::Array{ComplexF64,4}, U::Array{ComplexF64,3},
    w_list::Vector{Float64}, kplusb::Matrix{Int64}, Nk::Int64, Nb::Int64,
    Ne::Int64, omega::Vector{Float64}, grad_omega::Array{ComplexF64,3}, obj_only::Bool)

    @ccall lib_path.__oracles_MOD_omega_oracle(S::Ptr{ComplexF64}, R::Ptr{ComplexF64}, U::Ptr{ComplexF64},
        w_list::Ptr{Float64}, kplusb::Ptr{Int64}, Nk::Ref{Int64}, Nb::Ref{Int64},
        Ne::Ref{Int64}, omega::Ref{Float64}, grad_omega::Ptr{ComplexF64}, obj_only::Ref{Bool})::Nothing
    return omega
end
 =#

# f_oracle(S, Rmn, U, w, kplusb, rho_hat, Nk, Nb, Ne, Nj, omega)
function f_oracle!(S::Array{ComplexF64,4}, R::Array{ComplexF64,4}, U::Array{ComplexF64,3},
    w_list::Vector{Float64}, kplusb::Matrix{Int64}, rho_hat::Matrix{ComplexF64}, Nk::Int64, Nb::Int64,
    Ne::Int64, Nj::Int64, omega::Vector{Float64})

    @ccall lib_path.__oracles_MOD_f_oracle(S::Ptr{ComplexF64}, R::Ptr{ComplexF64}, U::Ptr{ComplexF64},
        w_list::Ptr{Float64}, kplusb::Ptr{Int64}, rho_hat::Ptr{ComplexF64}, Nk::Ref{Int64}, Nb::Ref{Int64},
        Ne::Ref{Int64}, Nj::Ref{Int64}, omega::Ref{Float64})::Nothing
    return omega
end


# grad_f_oracle(Rmn, w, rho_hat, Nk, Nb, Ne, Nj, grad_omega)
function grad_f_oracle!(R::Array{ComplexF64,4}, w_list::Vector{Float64}, rho_hat::Matrix{ComplexF64}, Nk::Int64, Nb::Int64,
    Ne::Int64, Nj::Int64, grad_omega::Array{ComplexF64,3})

    @ccall lib_path.__oracles_MOD_grad_f_oracle(R::Ptr{ComplexF64}, w_list::Ptr{Float64},
        rho_hat::Ptr{ComplexF64}, Nk::Ref{Int64}, Nb::Ref{Int64}, Ne::Ref{Int64}, Nj::Int64,
        grad_omega::Ref{ComplexF64})::Nothing
end


# project(U, grad_omega, grad_work, Nk, Ne)
function project!(U::Array{ComplexF64,3}, grad_omega::Array{ComplexF64,3},
    grad_work::Array{ComplexF64,2}, Nk::Int64, Nb::Int64)

    @ccall lib_path.__oracles_MOD_project(U::Ptr{ComplexF64}, grad_omega::Ptr{ComplexF64},
        grad_work::Ptr{ComplexF64}, Nk::Ref{Int64}, Nb::Ref{Int64})::Nothing
end

# retract(U, DeltaU, U_work, Nk, Ne, ideg, size_u_work)
function retract!(U::Array{ComplexF64,3}, delta_U::Array{ComplexF64,3},
    U_work::Vector{ComplexF64}, Nk::Int64, Nb::Int64, ideg::Int64, size_u_work::Int64)

    @ccall lib_path.__oracles_MOD_retract(U::Ptr{ComplexF64}, delta_U::Ptr{ComplexF64},
        U_work::Ptr{ComplexF64}, Nk::Ref{Int64}, Nb::Ref{Int64},
        ideg::Ref{Int64}, size_u_work::Ref{Int64})::Nothing
end
