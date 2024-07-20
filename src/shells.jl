using ProgressMeter
using DataStructures

export ApproximationScheme,
    CosScheme,
    CosScheme3D,
    find_shells,
    compute_weights,
    spread,
    shells,
    weights,
    populate_integral_table!,
    center,
    second_moment,
    spread,
    neighbor_basis_integral,
    BranchStable,
    W90BranchCut,
    TruncatedConvolution,
    STDC,
    RTDC,
    gauge_gradient,
    scheme,
    gauge_gradient_k_point_contribution


"""
    ApproximationScheme

The center and the spread are often approximated for performance.
Concrete approximations should subtype this.
"""
abstract type ApproximationScheme end

"""
    CosScheme

Approximations based on a single mode of cos.
"""
abstract type CosScheme <: ApproximationScheme end

struct CosScheme3D <: CosScheme
    neighbor_shells::AbstractVector{Vector{<:KPoint}}
    weights::AbstractVector{Number}
    neighbor_basis_integral::NeighborIntegral
end

"""
    CosScheme3D(ũ, n_shells = 1)

The function ``r^2`` can be approximated with ``w_{\\mathbf{b}} \\cos(\\mathbf{b}^T
\\mathbf{r})`` functions, which are then used for approximating the convolution
between ``r^2`` and `u` (inverse fft of `ũ`).
The `CosScheme3D` includes shells of `\\mathbf{b}` vectors and their 
corresponding weights ``w_{\\mathbf{b}}``.

Other dimensions will be implemented as extensions.

# Example: 

```jldoctest orbital_set
julia> scheme = CosScheme3D(ũ);

julia> length(shells(scheme))
1
```
"""
function CosScheme3D(u::OrbitalSet{UnkBasisOrbital{ReciprocalLattice3D}}, n_shells = 1; integrate=true)
    neighbor_shells = find_shells(grid(u), n_shells)
    weights = compute_weights(neighbor_shells)
    weights === nothing && return CosScheme3D(u, n_shells + 1; integrate=integrate)
    neighbor_integral = NeighborIntegral()

    scheme = CosScheme3D(neighbor_shells, weights, neighbor_integral)
    integrate && populate_integral_table!(scheme, u)
    return scheme
end


"""
    shells(scheme)

Shells of ``\\mathbf{b}`` vectors involved in the approximation scheme.  Each
shell is a vector of kpoints.

Example: 

```jldoctest orbital_set
julia> shells(scheme)

1-element Vector{Vector{<:KPoint}}:
 KPoint[GridVector{BrillouinZone3D}:
    coefficients: [-1, -1, -1]
, GridVector{BrillouinZone3D}:
    coefficients: [-1, 0, 0]
, GridVector{BrillouinZone3D}:
    coefficients: [0, -1, 0]
, GridVector{BrillouinZone3D}:
    coefficients: [0, 0, -1]
, GridVector{BrillouinZone3D}:
    coefficients: [0, 0, 1]
, GridVector{BrillouinZone3D}:
    coefficients: [0, 1, 0]
, GridVector{BrillouinZone3D}:
    coefficients: [1, 0, 0]
, GridVector{BrillouinZone3D}:
    coefficients: [1, 1, 1]
]
```
"""
shells(scheme::ApproximationScheme)::AbstractVector{Vector{<:KPoint}} =
    scheme.neighbor_shells

"""
    weights(scheme)

The weights corresponding to each shell within a scheme. The weights are ordered
from the inner-most to the outer-most shell.

```jldoctest orbital_set
julia> weights(scheme)
1-element Vector{Number}:
 5.336038037571918
```
"""
weights(scheme::ApproximationScheme) = scheme.weights

"""
    neighbor_basis_integral(scheme)

The integrals between neighboring k-points (The MMN matrix). 
The integral is amonst immediate neighbor because the ``\\cos`` approximation
is truncated at the first mode.

```jldoctest orbital_set
julia> M = neighbor_basis_integral(scheme)
julia> M[brillouin_zone[0, 0, 0], brillouin_zone[0, 0, 1]]
4×4 Matrix{ComplexF64}:
     0.85246+0.512198im     0.0798774-0.00342381im  4.54441e-8-1.98313e-8im  8.32139e-9+4.06675e-8im
   0.0057819+0.021992im     -0.363241-0.324958im     -0.393272-0.187671im     -0.489628+0.127022im
   0.0112149-0.000481798im  -0.194858+0.141126im      0.665299+0.0837434im    -0.498015-0.0113525im
 -0.00992131-0.0216156im     0.432817+0.269315im     -0.319281+0.193723im     -0.524017-0.0193465im
```
"""
neighbor_basis_integral(scheme::ApproximationScheme) = scheme.neighbor_basis_integral
# transformed_integral(scheme::ApproximationScheme) = scheme.transformed_integral

"""
    find_neighbors(k, scheme)

Find the set of relevant neighbors for a kpoint under a scheme.
"""
function find_neighbors(kpoint::KPoint, scheme::ApproximationScheme)
    dk_list = vcat(shells(scheme)...)
    # TODO: The negative part may not be necessary.
    return (dk -> kpoint + dk).([dk_list; -dk_list])
end

# function gauge_transform(scheme::ApproximationScheme, U::Gauge)
#     @set scheme.neighbor_basis_integral = gauge_transform(neighbor_basis_integral(scheme), U)
# end


"""
The Brillouin zone on which the finite difference scheme is defined.
"""
WTP.grid(scheme::ApproximationScheme) = grid(shells(scheme)[1][1])

function find_shells(grid::Grid, n_shell::Int)
    shells = SortedDict{Real,Vector{KPoint}}()
    d = collect(-2*n_shell:2*n_shell)
    for i in d, j in d, k in d
        k = make_grid_vector(grid, [i, j, k])
        key = round(norm(coordinates(k)), digits = 5)
        haskey(shells, key) ? append!(shells[key], [k]) : shells[key] = [k]
    end

    return collect(values(shells))[2:n_shell+1]
end

"""
Solve Aw = q
"""
function compute_weights(neighbor_shells::Vector{Vector{T}}) where {T<:AbstractGridVector}

    indices = SortedDict(
        (1, 1) => 1,
        (1, 2) => 2,
        (1, 3) => 3,
        (2, 2) => 4,
        (2, 3) => 5,
        (3, 3) => 6,
    )

    A = zeros((6, length(neighbor_shells)))
    c(b, i) = coordinates(b)[i]
    for s = 1:length(neighbor_shells)
        A[:, s] =
            [sum((b -> c(b, i) * c(b, j)).(neighbor_shells[s])) for (i, j) in keys(indices)]
    end

    q = [i == b ? 1 : 0 for (i, b) in keys(indices)]
    w = A \ q

    return isapprox(A * w, q, atol = 1e-5) ? w : nothing
end

function populate_integral_table!(scheme::ApproximationScheme, u::OrbitalSet)
    brillouin_zone = grid(u)
    M = neighbor_basis_integral(scheme)

    """
    Compute the mmn matrix between k1 and k2.
    """
    function mmn_matrix(k_1::T, k_2::T) where {T<:AbstractGridVector{<:BrillouinZone}}
        U = hcat(vectorize.(u[k_1])...)
        V = hcat(vectorize.(u[k_2])...)
        return adjoint(U) * V
        # return [braket(m, n) for m in u[k_1], n in u[k_2]]
    end

    @showprogress for k in collect(brillouin_zone)
        for neighbor in find_neighbors(k, scheme)
            # M[k, neighbor] = adjoint(U[k]) * mmn_matrix(k, neighbor) * U[neighbor]
            haskey(M, k, neighbor) && continue
            M[k, neighbor] = mmn_matrix(k, neighbor)
        end
    end
    return M
end


