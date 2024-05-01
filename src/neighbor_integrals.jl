export NeighborIntegral, integrals, gauge_transform

"""
The neighbor integrals indexed by two kpoints.  For each pair of kpoint, the
integrals are stored as a matrix. This is the same as the ``M_{mn}^{k, b}``
matrix in MLWF. The matrix elements are accessed by `M[k, k+b][m, n]`
"""
struct NeighborIntegral
    integrals::Dict{Pair{KPoint,KPoint},Matrix{ComplexFxx}}
end

"""
    NeighborIntegral() 

The neighbor integrals is roughly a dictionary with pairs of neighboring
k-points as the keys. One can create one just by
"""
NeighborIntegral() = NeighborIntegral(Dict())
integrals(n::NeighborIntegral) = n.integrals

function Base.hash(p::Pair{<:KPoint,<:KPoint})
    m, n = p
    a_nice_prime_number = 7
    return hash(m) * a_nice_prime_number + hash(n)
end

function Base.:(==)(p_1::Pair{<:KPoint,<:KPoint}, p_2::Pair{<:KPoint,<:KPoint})
    m_1, n_1 = p_1
    m_2, n_2 = p_2
    return m_1 == m_2 && n_1 == n_2
end

"""
    getindex(M, k_1, k_2)

The integral matrix between the neighboring k-points `k_1` and `k_2`.
Can also write `M[k_1, k_2]`. Note that `M[k_1, k_2] = M[k_2, k_1]'`.
So only one of the matrices is stored.

```jldoctest orbitla_set
julia> M[brillouin_zone[0, 0, 0], brillouin_zone[0, 0, -1]] == M[brillouin_zone[0, 0, -1], brillouin_zone[0, 0, 0]]'
true
```
"""
function Base.getindex(neighbor_integral::NeighborIntegral, k_1::KPoint, k_2::KPoint)
    # coefficients(k_1) == coefficients(k_2) && return I
    i = integrals(neighbor_integral)
    # haskey(i, k_1 => k_2) && return i[k_1=>k_2]
    # return adjoint(i[k_2=>k_1])
    result = get(i, k_1 => k_2, nothing)
    return result === nothing ? adjoint(i[k_2=>k_1]) : result
end

"""
    setindex!(M, value, g...)

Set the integral matrix between two k-points `g[1]` and `g[2]` to
`value`. Can also write `M[g...] = value`.
"""
function Base.setindex!(neighbor_integral::NeighborIntegral, value::AbstractMatrix, g::Vararg{<:KPoint})
    g_1, g_2 = g
    integrals(neighbor_integral)[g_1=>g_2] = value
end

"""
    haskey(M, k_1, k_2)

Check if the integral matrix between `k_1` and `k_2` has been computed and stored.

```jldoctest orbital_set
julia> haskey(M, brillouin_zone[0, 0, 1], brillouin_zone[0, 0, 0])
true

julia> haskey(M, brillouin_zone[0, 0, 1], brillouin_zone[0, 0, -1])
false
```
"""
function Base.haskey(neighbor_integral::NeighborIntegral, k_1::KPoint, k_2::KPoint)
    i = integrals(neighbor_integral)
    return haskey(i, k_1 => k_2) || haskey(i, k_2 => k_1)
end

"""
    gauge_transform(M, gauge)

Perform a gauge transform on the neighbor integrals.

``U^{k \\dagger} M^{k, k+b} U^{k+b}``

```jldoctest orbital_set
julia> M = gauge_transform(M, U);
julia> M[brillouin_zone[0, 0, 1], brillouin_zone[0, 0, 0]]
4×4 adjoint(::Matrix{ComplexF64}) with eltype ComplexF64:
    0.85246-0.512198im    0.0057819-0.021992im  0.0112149+0.000481798im  -0.00992131+0.0216156im
  0.0798774+0.00342381im  -0.363241+0.324958im  -0.194858-0.141126im        0.432817-0.269315im
 4.54441e-8+1.98313e-8im  -0.393272+0.187671im   0.665299-0.0837434im      -0.319281-0.193723im
 8.32139e-9-4.06675e-8im  -0.489628-0.127022im  -0.498015+0.0113525im      -0.524017+0.0193465im
```

"""
function gauge_transform(neighbor_integral::NeighborIntegral, gauge::Gauge)
    t = NeighborIntegral()
    for ((k_1, k_2), integral) in integrals(neighbor_integral)
        t[k_1, k_2] = adjoint(gauge[k_1]) * integral * gauge[k_2]
    end
    return t
end


"""
    NeighborIntegral(mmn, k_map)

Construct a `NeighborIntegral` from a `MMN` object and a `k_map`.

We like to convert this raw data to `NeighborIntegral` before looking up
integrals from it. One problem that we encounter is that the k-points are stored
as a single integer in the `.mmn` files. To find out which k-points these
integers correspond to, we have to construct a mapping using the `wfcX.dat`
files (use `i_kpoint_map`).

```@jldoctest wfc
julia> neighbor_integral = NeighborIntegral(mmn, k_map);
julia> neighbor_integral[brillouin_zone[0, 0, 0], brillouin_zone[1, 0, 0]][:, :]
4×4 Matrix{ComplexF64}:
   0.766477+0.633678im    0.0796606-0.00680469im  -3.64646e-7-3.84398e-7im   4.348e-9-2.891e-8im
 -0.0205239-0.0122884im    0.498436-0.12016im        0.127234-0.327676im     0.121905+0.521335im
 -0.0158759-0.0144468im    0.459743-0.0173616im      -0.52219+0.121039im    -0.257164-0.358587im
 -0.0131809-0.00142574im   0.223027-0.176081im       0.606245-0.0237135im   0.0852059-0.53885im
```
"""
function NeighborIntegral(mmn::MMN, k_map::Dict{Int64,KPoint})
    n = NeighborIntegral()

    for ((i_kpoint1, i_kpoint2), integral) in mmn.integrals
        kpoint = k_map[i_kpoint1]
        neighbor = add_overflow(k_map[i_kpoint2], mmn.translations[i_kpoint1=>i_kpoint2])
        neighbor = add_overflow(neighbor, -overflow(kpoint))
        kpoint = reset_overflow(kpoint)
        n[kpoint, neighbor] = integral
    end
    return n
end
