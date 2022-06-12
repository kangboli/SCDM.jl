using Documenter, WTP, SCDM, LinearAlgebra

push!(LOAD_PATH,"../src/")
makedocs(sitename="SCDM.jl",
modules=[SCDM],
checkdocs=:none,
doctest=false,
pages = [
    "Home" => "index.md",
    "Center & Spread" => "center_spread.md",
    "SCDM" => "scdm.md",
    "Manopt " => "manopt.md",
    # "Center and Spread" => "center_spread.md"
]
)

# deploydocs(
#     repo = "github.com/kangboli/WTP.jl",
# )