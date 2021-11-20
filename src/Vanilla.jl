export scdm_condense_phase

function scdm_condense_phase(wannier::Wannier{UnkBasisOrbital{HomeCell}}, bands::AbstractVector{Int}, ortho=true)
    brillouin_zone = grid(wannier)
    gamma_point = brillouin_zone[0, 0, 0]
    homecell = grid(wannier[gamma_point][1])

    vectorize(o::UnkBasisOrbital{HomeCell}) = reshape(elements(o), prod(size(grid(o))))

    Ψ_Γ = hcat(vectorize.(wannier[gamma_point][bands])...)

    F = qr(Ψ_Γ', Val(true))
    permutation = F.p

    columns = permutation[1:length(bands)]
    # println(columns)

    gauge = Gauge(brillouin_zone)

    orthonormalize(A::AbstractMatrix) = let (U, _, V) = svd(A)
        U * adjoint(V)
    end

    @showprogress for k in collect(brillouin_zone)
        phase = (r->exp(1im * (r' * k))).(homecell[columns])
        Ψ = hcat(vectorize.(wannier[k][bands])...)

        Ψ = diagm(phase) * Ψ[columns, :]
        Ψ = Ψ'

        gauge[k] = ortho ? orthonormalize(Ψ) : Ψ
    end

    return gauge
end


        # if (k == brillouin_zone[0,0,1])
        #     for c in columns
        #         println("Real", coefficients(homecell[c]))
        #         println("Real", coefficients(homecell[c])./size(homecell))
        #     end
        #     # println([(homecell[c]' * k) / (2 * pi) for c in columns]) 
        #     # println(phase)
        # end