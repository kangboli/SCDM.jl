export scdm_condense_phase

"""
    scdm_condense_phase(u, bands, [ortho=true])

Perform SCDM for condense phase orbitals.
The `bands` amongst the orbitals `u::Wannier` will be localized.
"""
function scdm_condense_phase(wannier::Wannier{UnkBasisOrbital{T}}, bands::AbstractVector{<:Integer}, ortho=true) where T <: HomeCell
    brillouin_zone = grid(wannier)
    gamma_point = brillouin_zone[0, 0, 0]
    homecell = grid(wannier[gamma_point][1])

    vectorize(o::UnkBasisOrbital{T}) = reshape(elements(o), prod(size(grid(o))))

    Ψ_Γ = hcat(vectorize.(wannier[gamma_point][bands])...)

    F = qr(Ψ_Γ', Val(true))
    permutation = F.p

    columns = permutation[1:length(bands)]
    # println(columns)

    n_bands_complete = length(elements(wannier)[1])
    U = Gauge(brillouin_zone, n_bands_complete)

    orthonormalize(A::AbstractMatrix) = let (U, _, V) = svd(A)
        U * adjoint(V)
    end

    @showprogress for k in collect(brillouin_zone)
        phase = (r->exp(1im * (r' * k))).(homecell(columns))
        Ψ = hcat(vectorize.(wannier[k][bands])...)

        Ψ = diagm(phase) * Ψ[columns, :]
        Ψ = Ψ'

        U[k][bands, bands] = ortho ? orthonormalize(Ψ) : Ψ
    end

    return U
end


        # if (k == brillouin_zone[0,0,1])
        #     for c in columns
        #         println("Real", coefficients(homecell[c]))
        #         println("Real", coefficients(homecell[c])./size(homecell))
        #     end
        #     # println([(homecell[c]' * k) / (2 * pi) for c in columns]) 
        #     # println(phase)
        # end