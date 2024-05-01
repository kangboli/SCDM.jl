"""
Use the same algorithm as in the original Wannier90 (without guiding centers,
fixed centers, etc.).  Apply this algorithm if you like to cross validate with
Wannier90.
"""
abstract type W90BranchCut end
"""
Mostly the same algorithms as `W90BranchCut`, but make a potential more
consistent choice of branch cut.
"""
abstract type BranchStable end
"""
The truncated convolution algorithm. (Resta's first spread)
"""
abstract type TruncatedConvolution end

"""
The separated truncated convolution. 
"""
abstract type STDC end

"""
The reduced truncated convolution
"""
abstract type RTDC end

center(scheme::CosScheme, n::Integer) = center(scheme, n, W90BranchCut)
center(scheme::CosScheme, n::Integer, ::Type{T}) where {T} = center(neighbor_basis_integral(scheme), scheme, n, T)
center(M::NeighborIntegral, scheme::CosScheme, n::Integer) = center(M, scheme, n, W90BranchCut)

"""
    center(M, scheme, n, ::Type{W90BranchCut})

Compute the center of the `n`th Wannier orbital using the original Wannier90 approach.
The replication is exact.

Example:

```jldoctest orbital_set
julia> center(M, scheme, 1, W90BranchCut)
3-element Vector{Float64}:
 -8.73495454038011
  3.9151488754936588
  4.021344063664258
julia> center(M, scheme, 2, W90BranchCut)
3-element Vector{Float64}:
  0.11705358388655285
 -1.4773864607361118
 -2.4684488265569624
```
"""
function center(M::NeighborIntegral, scheme::CosScheme, n::Int, ::Type{W90BranchCut})
    kpoint_contribution(k::KPoint) = -sum(zip(weights(scheme), shells(scheme))) do (w, shell)
        sum(b -> w * coordinates(b) * angle(M[k, k+b][n, n]), shell)
    end

    brillouin_zone = collect(grid(scheme))
    return sum(kpoint_contribution.(brillouin_zone)) / length(brillouin_zone)
end

"""
    center(M, scheme, n, ::Type{TruncatedConvolution})

Compute the center of the `n`th Wannier orbital using the turncated convolution algorihtm.

Example:

```jldoctest orbital_set
julia> center(M, scheme, 1, TruncatedConvolution)
3-element Vector{Float64}:
 -8.70234629129622
  4.016099860574682
  4.093053970246974
julia> center(M, scheme, 1, TruncatedConvolution)
3-element Vector{Float64}:
  1.3634042350006577
 -2.712474285697622
 -3.6252083684030336
```

One can compare this with the "exact" center

```jldoctest orbital_set
julia> wanniers = commit_gauge(ũ)(:);
julia> _, r̃2 = compute_r2(supercell(u));
julia> c_1, σ²_1 = center_spread(fft(abs2(ifft(wanniers[1])), false), r̃2)
([-8.831796622851812, 3.953063696051289, 3.940388714312329], 28.748708554079474)
julia> c_2, σ²_2 = center_spread(fft(abs2(ifft(wanniers[2])), false), r̃2)
([-9.192231000846695, 7.604578700842824, 6.189389913055544], 27.460489592320883)
```
"""
function center(M::NeighborIntegral, scheme::CosScheme, n::Int, ::Type{TruncatedConvolution})
    brillouin_zone = collect(grid(scheme))

    phase(b::KPoint) = angle(1 / length(brillouin_zone) * sum(k -> M[k, k+b][n, n], brillouin_zone))

    return -sum(zip(weights(scheme), shells(scheme))) do (w, shell)
        sum(unique(k -> Set([k, -k]), shell)) do b
            ϕ⁺, ϕ⁻ = phase(b), phase(-b)
            branch = (sign(ϕ⁺) == sign(ϕ⁻) ? -1 : 1)
            w * coordinates(b) * ϕ⁺ + w * cartesian(-b) * branch * ϕ⁻
        end
    end
end

function center(M::NeighborIntegral, scheme::CosScheme, n::Int, ::Type{BranchStable})
    kpoint_contribution(k::KPoint) = -sum(zip(weights(scheme), shells(scheme))) do (w, shell)
        sum(unique(k -> Set([k, -k]), shell)) do b
            ϕ⁺ = M[k, k+b][n, n] |> angle
            ϕ⁻ = M[k, k-b][n, n] |> angle
            branch = (sign(ϕ⁺) == sign(ϕ⁻) ? -1 : 1)
            w * coordinates(b) * ϕ⁺ + w * cartesian(-b) * branch * ϕ⁻
        end
    end

    brillouin_zone = scheme |> grid |> collect
    return sum(kpoint_contribution.(brillouin_zone)) / prod(size(brillouin_zone))

end

second_moment(scheme::CosScheme, n::Int) = second_moment(neighbor_basis_integral(scheme), scheme, n)

function second_moment(M::NeighborIntegral, scheme::CosScheme, n::Int)
    kpoint_contribution(k::KPoint) =
        sum(zip(weights(scheme), shells(scheme))) do (w, shell)
            sum(b -> w * (1 - abs2(M[k, k+b][n, n]) + angle(M[k, k+b][n, n])^2), shell)
        end
    brillouin_zone = collect(grid(scheme))
    return sum(kpoint_contribution.(brillouin_zone)) / prod(size(brillouin_zone))
end

spread(scheme::CosScheme, n::Integer) = spread(scheme, n, W90BranchCut)
spread(scheme::CosScheme, n::Integer, ::Type{T}) where {T} = spread(neighbor_basis_integral(scheme), scheme, n, T)
spread(M::NeighborIntegral, scheme::CosScheme, n::Integer) = spread(M, scheme, n, W90BranchCut)

"""
    spread(M, scheme, n, W90BranchCut)

Compute the spread of the `n`th Wannier orbital using the original Wannier90 approach.

```jldoctest orbitla_set
julia> spread(M, scheme, 1, W90BranchCut)
15.742034477681969
julia> spread(M, scheme, 2, W90BranchCut) # failure.
133.09413338071354
```
"""
spread(M::NeighborIntegral, scheme::CosScheme, n::Integer, ::Type{T}) where {T} =
    second_moment(M, scheme, n) - norm(center(M, scheme, n, T))^2

"""
    spread(M, scheme, n, TruncatedConvolution)

Compute the spread of the `n`th Wannier orbital using the truncated convolution.

```jldoctest orbitla_set
julia> spread(M, scheme, 1, TruncatedConvolution)
17.50438313709964
julia> spread(M, scheme, 2, TruncatedConvolution)
17.313972338201154
```

Compare this to the "exact" spread.

```jldoctest orbital_set 
julia> σ²_1, σ²_2
(28.748708554079474, 27.460489592320883)
```
"""
function spread(M::NeighborIntegral, scheme::CosScheme, n::Integer, ::Type{TruncatedConvolution})
    brillouin_zone = collect(grid(scheme))
    ρ̃(b) = sum(k -> M[k, k+b][n, n], brillouin_zone) / length(brillouin_zone)

    sum(zip(weights(scheme), shells(scheme))) do (w, shell)
        sum(b -> 2w * (1 - abs(ρ̃(b))), shell)
    end
end

function spread(M::NeighborIntegral, scheme::CosScheme, n::Integer, ::Type{STDC})
    brillouin_zone = collect(grid(scheme))
    ρ̃(b) = sum(k -> abs(M[k, k+b][n, n]), brillouin_zone) / length(brillouin_zone)

    sum(zip(weights(scheme), shells(scheme))) do (w, shell)
        sum(b -> 2w * (1 - abs(ρ̃(b))), shell)
    end
end
function spread(M::NeighborIntegral, scheme::CosScheme, n::Integer, ::Type{RTDC})
    brillouin_zone = collect(grid(scheme))
    ρ̃(b) = sum(k -> M[k, k+b][n, n], brillouin_zone) / length(brillouin_zone)
    shell = [grid(scheme)[0,0,1], grid(scheme)[0,1,0], grid(scheme)[1,0,0],
            grid(scheme)[0,0,-1], grid(scheme)[0,-1,0], grid(scheme)[-1,0,0]
            ]
    let w = weights(scheme)[1]
        for b in shell
            println(ρ̃(b))
            println(abs(ρ̃(b)))
        end
        sum(b -> 2w * (1 - abs(ρ̃(b))), shell)
    end
end

# for k in brillouin_zone
#     (m -> cache!(m, M)).(u[k])
# end

function WTP.n_band(M::NeighborIntegral)
    first_matrix = collect(values(integrals(M)))[1]
    return size(first_matrix, 1)
end

gauge_gradient(M::NeighborIntegral, scheme::CosScheme, brillouin_zone::BrillouinZone) =
    gauge_gradient(M, scheme, brillouin_zone, W90BranchCut)


"""
    gauge_gradient(M, scheme, brillouin_zone, W90BranchCut)

Gauge gradient in the original Wannier90. 
The result is a `OnGrid{<:BrillouinZone}`
with a matrix on each k-point.

```jldoctest orbital_set
julia> G_w = gauge_gradient(M, scheme, brillouin_zone, W90BranchCut);
julia> G_w[brillouin_zone[0, 0, 1]]
4×4 Matrix{ComplexF64}:
       0.0-0.030634im     -1.13757+0.493027im   0.145804-0.00940629im  -0.0627567-0.0302657im
   1.13757+0.493027im          0.0+1.76773im   -0.876742+0.121255im      0.885767+0.370595im
 -0.145804-0.00940629im   0.876742+0.121255im        0.0-0.158586im    -0.0557475+0.0957635im
 0.0627567-0.0302657im   -0.885767+0.370595im  0.0557475+0.0957635im          0.0-0.12397im
```
"""
function gauge_gradient(M::NeighborIntegral, scheme::CosScheme, brillouin_zone::B, ::Type{W90BranchCut}) where {B<:BrillouinZone}
    N = n_band(M)
    c = (n -> center(M, scheme, n, W90BranchCut)).(1:N)

    G = k -> sum(zip(weights(scheme), shells(scheme))) do (w, shell)
        sum(shell) do b
            A = M[k, k+b]
            q = [angle(A[n, n]) + coordinates(b)' * c[n] for n = 1:N]
            R = hcat([A[:, n] * A[n, n]' for n = 1:N]...)
            T = hcat([(A[:, n] / A[n, n]) * q[n] for n = 1:N]...)
            4w * ((R - R') / 2 - (T + T') / 2im) / length(brillouin_zone)
        end
    end

    return map(G, brillouin_zone)
end

"""
    gauge_gradient(M, scheme, brillouin_zone, TruncatedConvolution) 

Gauge gradient for the truncated convolution. 
The result is a `OnGrid{<:BrillouinZone}`
with a matrix on each k-point.

```jldoctest orbital_set
julia> G_t = gauge_gradient(U, scheme, brillouin_zone, TruncatedConvolution);
julia> G_t[brillouin_zone[0, 0, 1]]
4×4 Matrix{ComplexF64}:
        0.0-0.0388009im  -0.00321511+0.158118im    0.160808-0.0451399im  -0.0646812+0.0334942im
 0.00321511+0.158118im           0.0-0.179264im   -0.158035-0.0368048im    0.205487+0.0206957im
  -0.160808-0.0451399im     0.158035-0.0368048im        0.0-0.108496im    -0.070136+0.157672im
  0.0646812+0.0334942im    -0.205487+0.0206957im   0.070136+0.157672im          0.0-0.0640979im
```
"""
function gauge_gradient(M::NeighborIntegral, scheme::CosScheme, brillouin_zone::B, ::Type{TruncatedConvolution}) where {B<:BrillouinZone}
    # M = gauge_transform(neighbor_basis_integral(scheme), U)
    N = n_band(M)

    ρ̃ = Dict(b => (n -> 1 / N * sum(k -> M[k, k+b][n, n], brillouin_zone)).(1:N) for b in vcat(shells(scheme)...))

    G = k -> sum(zip(weights(scheme), shells(scheme))) do (w, shell)
        sum(shell) do b
            R = hcat([M[k, k+b][:, n] * ρ̃[b][n]' / abs(ρ̃[b][n]) for n = 1:N]...)
            2w * (R - R') / length(brillouin_zone)
        end
    end 

    return map(G, brillouin_zone)
end

function gauge_gradient(M::NeighborIntegral, scheme::CosScheme, brillouin_zone::B, ::Type{RTDC}) where {B<:BrillouinZone}
    # M = gauge_transform(neighbor_basis_integral(scheme), U)
    N = n_band(M)

    ρ̃ = Dict(b => (n -> 1 / N * sum(k -> M[k, k+b][n, n], brillouin_zone)).(1:N) for b in vcat(shells(scheme)...))

    G = k -> let w = weights(scheme)[1]
        sum([brillouin_zone[0,0,1], brillouin_zone[0,1,0], brillouin_zone[1,0,0],
        brillouin_zone[0,0,-1], brillouin_zone[0,-1,0], brillouin_zone[-1,0,0]
        ]) do b
            R = hcat([M[k, k+b][:, n] * ρ̃[b][n]' / abs(ρ̃[b][n]) for n = 1:N]...)
            2w * (R - R') / length(brillouin_zone)
        end
    end 

    return map(G, brillouin_zone)
end


function gauge_gradient_k_point_contribution(M::NeighborIntegral, scheme::CosScheme, brillouin_zone::B, k::KPoint, ::Type{STDC}) where {B<:BrillouinZone}
    # M = gauge_transform(neighbor_basis_integral(scheme), U)
    N = n_band(M)

    sum(zip(weights(scheme), shells(scheme))) do (w, shell)
        sum(shell) do b
            R = hcat([M[k, k+b][:, n] * M[k, k+b][n, n]' / abs(M[k, k+b][n, n]) for n = 1:N]...)
            2w * (R - R') / length(brillouin_zone)
        end
    end
end

function gauge_gradient(M::NeighborIntegral, scheme::CosScheme, brillouin_zone::B, ::Type{STDC}) where {B<:BrillouinZone}
    map(k->gauge_gradient_k_point_contribution(M, scheme, brillouin_zone, k, STDC), brillouin_zone)
end

#= function all_spread_k_point_contribution(M::NeighborIntegral, scheme, k, ::Type{Branch}) where {Branch}
    # M = gauge_transform(neighbor_basis_integral(scheme), U)
    N = n_band(M)
    return map(1:N) do n
        spread_k_point_contribution(M, scheme, n, k, Branch)
    end
end
 =#
