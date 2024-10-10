module SCDM

using WTP
using LinearAlgebra

include("selected_columns.jl")
include("neighbor_integrals.jl")
include("shells.jl")
include("slow_oracles.jl")
include("oracles.jl")
include("gpu_oracles.jl")
include("flatten.jl")
include("conjugate_gradient.jl")
include("oracles_mv.jl")
#= include("oracle_wrapper.jl") =#
#= include("Manopt.jl") =#
#= include("AmateurOptimizers.jl") =#

end
