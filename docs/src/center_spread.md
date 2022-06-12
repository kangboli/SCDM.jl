# Center and Spread

## Approximation Schemes

```@docs
ApproximationScheme
```

```@docs
CosScheme
```

```@docs
CosScheme3D(::OrbitalSet{UnkBasisOrbital{ReciprocalLattice3D}})
```

```@docs
shells
```

```@docs
weights
```

## The neighbor integrals

### Create a neighbor integral.

```@dosc
find_neighbors
```

```@docs
NeighborIntegral
```

```@docs
neighbor_basis_integral
```

### Indexing the neighbor integral.

```@docs
getindex(::NeighborIntegral, ::KPoint, ::KPoint)
```

```@docs
setindex!(::NeighborIntegral, ::AbstractMatrix, ::Vararg{<:KPoint})
```

```@docs
haskey(::NeighborIntegral, ::KPoint, ::KPoint)
```

### Gauge transform

```@docs
gauge_transform
```

## Center and Spread

### Two sets of algorithms

```@docs
W90BranchCut
BranchStable
TruncatedConvolution
```

### Centers and spreads

```@docs
SCDM.center(::NeighborIntegral, ::CosScheme, ::Int, ::Type{W90BranchCut})
```

```@docs
SCDM.center(::NeighborIntegral, ::CosScheme, ::Int, ::Type{TruncatedConvolution})
```

```@docs
spread(::NeighborIntegral, ::CosScheme, ::Int, ::Type{W90BranchCut})
```

```@docs
spread(::NeighborIntegral, ::CosScheme, ::Int, ::Type{TruncatedConvolution})
```

## Spread gradient and minimization

```@docs
gauge_gradient(::NeighborIntegral, ::CosScheme, ::B, ::Type{W90BranchCut}) where {B<:BrillouinZone}
```

```@docs
gauge_gradient(::NeighborIntegral, ::CosScheme, ::B, ::Type{TruncatedConvolution}) where {B<:BrillouinZone}
```
