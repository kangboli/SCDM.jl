module SCDM

using WTP
using LinearAlgebra

include("Vanilla.jl")
include("NeighborIntegrals.jl")
include("CenterSpread.jl")
include("IterativeOptimizer.jl")

end
