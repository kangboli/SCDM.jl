using IterTools
export scdm_condense_phase, quadratic_pivot_scdm, box_grid, repack!, find_pivot

"""
    scdm_condense_phase(u, bands, [ortho=true])

Perform SCDM for condense phase orbitals.
The `bands` amongst the orbitals `u::OrbitalSet` will be localized.
The rest of the gauge will be an identity matrix.

The result will be a Gauge `U` and the selected `columns`.
"""
function scdm_condense_phase(u::OrbitalSet{UnkBasisOrbital{T}}, bands::AbstractVector{<:Integer}, ortho=true) where {T<:HomeCell}
    brillouin_zone, homecell = grid(u), orbital_grid(u)
    Γ = brillouin_zone[0, 0, 0]

    columns = begin
        Ψ_Γ = hcat(vectorize.(u[Γ][bands])...)
        F = qr(Ψ_Γ', ColumnNorm())
        F.p[1:length(bands)]
    end

    U = Gauge(brillouin_zone, n_band(u))

    orthonormalize(A::AbstractMatrix) =
        let (U, _, V) = svd(A)
            U * adjoint(V)
        end

    function normalize(A::AbstractMatrix)
        return hcat([c / norm(c) for c in eachcol(A)]...)
    end

    for k in collect(brillouin_zone)
        phase = (r -> exp(1im * (k' * r))).(homecell(columns))
        Ψ = hcat(vectorize.(u[k][bands])...)
        Ψ = diagm(phase) * Ψ[columns, :]
        Ψ = Ψ'

        U[k][bands, bands] = ortho ? orthonormalize(Ψ) : normalize(Ψ)
    end

    return U, columns
end

mutable struct Box
    mat::Matrix
    indices::Vector{Int}
    cursor::Int
    normsq::Vector{Float64}
    inverse_map::Vector{Int}
end

function box(ΨT, indices)
    Box(ΨT, indices, 1, zeros(size(ΨT, 2)), zeros(size(ΨT, 2)))
end

inverse_map(b::Box) = b.inverse_map

function pivot_with_norm(b::Box)
    i_pivot = argmax(b.normsq)
    return (i_pivot, b.normsq[i_pivot])
end

const neighbor_indices = collect(product(-1:1, -1:1, -1:1))

struct BoxGrid
    boxes::Array{Box}
    box_size::Int
    l_box::Int
end

boxes(b::BoxGrid) = b.boxes
l_box(b::BoxGrid) = b.l_box

function box_grid(l_grid::Int, box_size::Int, n_band::Int)
    l_box = Int(ceil(l_grid / box_size))
    boxes = Array{Box}(undef, l_box, l_box, l_box)
    for (x, y, z) in product(1:l_box, 1:l_box, 1:l_box)
        boxes[x, y, z] = box(
            zeros(ComplexF64, n_band, box_size^3), [x, y, z])
    end
    return BoxGrid(boxes, box_size, l_box)
end


function find_box(bg::BoxGrid, r::GridVector)
    standard_indices = miller_to_standard(size(grid(r)),
        tuple(coefficients(r)...), (0, 0, 0))
    box_index = [Int(ceil(c / bg.box_size)) for c in
                 standard_indices]
    return boxes(bg)[box_index...]
end


function repack!(Ψ_Γ::Matrix, bg::BoxGrid, homecell::HomeCell3D)
    for r in homecell
        b = find_box(bg, r)
        ir = linear_index(r)
        b.mat[:, b.cursor] = Ψ_Γ[ir, :]
        b.normsq[b.cursor] = norm(b.mat[:, b.cursor])^2
        inverse_map(b)[b.cursor] = ir
        b.cursor += 1
    end
    map(b -> b.cursor = 1, boxes(bg))
    bg
end

function find_pivot(bg::BoxGrid)
    sub_pivots = map(pivot_with_norm, boxes(bg))
    i_box = Tuple(argmax(last.(sub_pivots)))
    i_col, _ = sub_pivots[i_box...]
    return (i_box, i_col)
end

function mod_gram_schmidt!(b::Box, projector::Vector)
    for i in axes(b.mat, 2)
        overlap = projector' * b.mat[:, i]
        b.mat[:, i] -= overlap * projector
        b.normsq[i] -= abs2(overlap)
    end
end

function neighbors(bg::BoxGrid, i_box)
    result = []
    for d in neighbor_indices
        n = [d...] + [i_box...]
        (any(n .> l_box(bg)) || any(n .< 1)) && continue
        push!(result, boxes(bg)[n...])
    end
    return result
end


function quadratic_pivot_scdm(u::OrbitalSet{UnkBasisOrbital{T}}, bands::AbstractVector{<:Integer}, box_size::Int) where {T<:HomeCell}
    Γ = grid(u)[0, 0, 0]
    #= length(brillouin_zone) > 1 && error("only the gamma point formalism has been developed for quadratic time pivots") =#
    homecell = orbital_grid(u)
    bg = box_grid(x_max(homecell) - x_min(homecell) + 1, box_size, length(bands))

    Ψ_Γ = hcat(vectorize.(u[Γ][bands])...)
    repack!(Ψ_Γ, bg, homecell)
    columns = zeros(Int, length(bands))
    for i = 1:length(bands)
        i_box, i_col = find_pivot(bg)
        box = boxes(bg)[i_box...]
        columns[i] = inverse_map(box)[i_col]
        projector = box.mat[:, i_col] / sqrt(box.normsq[i_col])
        for b in neighbors(bg, i_box)
            mod_gram_schmidt!(b, projector)
        end
    end
    return columns
end
