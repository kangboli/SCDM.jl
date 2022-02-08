export scdm_condense_phase

"""
    scdm_condense_phase(u, bands, [ortho=true])

Perform SCDM for condense phase orbitals.
The `bands` amongst the orbitals `u::Wannier` will be localized.
The rest of the gauge will be an identity matrix.
"""
function scdm_condense_phase(u::OrbitalSet{UnkBasisOrbital{T}}, bands::AbstractVector{<:Integer}, ortho=true) where T <: HomeCell
    brillouin_zone, homecell = grid(u), orbital_grid(u)
    Γ = brillouin_zone[0, 0, 0]

    columns = begin
        Ψ_Γ = hcat(vectorize.(u[Γ][bands])...)
        F = qr(Ψ_Γ', ColumnNorm())
        F.p[1:length(bands)]
    end

    U = Gauge(brillouin_zone, n_band(u))

    orthonormalize(A::AbstractMatrix) = let (U, _, V) = svd(A)
        U * adjoint(V)
    end

    function normalize(A::AbstractMatrix) 
        return hcat([c / norm(c) for c in eachcol(A)]...)
    end

    for k in collect(brillouin_zone)
        phase = (r->exp(1im * (k' * r))).(homecell(columns))
        Ψ = hcat(vectorize.(u[k][bands])...)
        Ψ = diagm(phase) * Ψ[columns, :]
        Ψ = Ψ'

        U[k][bands, bands] = ortho ? orthonormalize(Ψ) : normalize(Ψ)
    end

    return U, columns
end
