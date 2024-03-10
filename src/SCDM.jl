module SCDM

using WTP
using LinearAlgebra

include("Vanilla.jl")
include("NeighborIntegrals.jl")
include("CenterSpread.jl")
include("Flatten.jl")
include("oracle_wrapper.jl")
#= include("Manopt.jl") =#
include("AmateurOptimizers.jl")

end
