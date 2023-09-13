#= PHOTOIONIZATION
   We basically reuse all the code in multigrid changing the applied stencil.
=#

"""
A single term in the Helmholtz approximation to photoionization.
"""
struct PhotoionizationTerm{T}
    "Multiplicative factor"
    a::T

    "k^2 in ∇²u - k² u = S"
    k2::T
end

"""
A set of terms representing a full photoionization model with `L` terms, affecting densities of two
species in `target`.
"""
struct PhotoionizationModel{L, T, BC, P}
    "Each `PhotoionizationTerm` in the Helmholtz approximation."
    term::SVector{L, PhotoionizationTerm{T}}

    "Species index of the source."
    source::Int
    
    "Target species (generally electrons and the ionized positive ions)"
    target::NTuple{2, Int}    

    "Boundary conditions for the photoionization fields."
    bc::BC

    "Number of vcycles in the solution."
    iter::Int
end

PhotoionizationModel{L, T, BC}(args...) where {L, T, BC} = PhotoionizationModel{L, T, BC, 2}(args...)
_order(::PhotoionizationModel{L, T, BC, P}) where {L, T, BC, P} = P

Base.length(::PhotoionizationModel{L}) where {L} = L

"""
Add the photoionization term to the density derivatives in `dni`.
"""
function photoionization!(tree::Tree{D}, dni, phfields, r, u1,
                          phmodel, h, conn, geom) where D
    (;term, source, target, bc, iter) = phmodel
    for i in eachindex(term)
        helm = HelmholtzDiscretization{D, _order(phmodel), typeof(term[i].k2)}(term[i].k2)
        for j in 1:iter
            vcycle!(phfields[i], dni[source], r, u1, h^2,
                    tree, conn, geom, bc, helm, nup=2, ndown=2, ntop=4)
        end
    end

    # Because the source may be changed, only after computing all fields can we add them to the
    # derivatives.
    for i in eachindex(term)
        a = term[i].a
        add_photoionization_term!(tree, dni, phfields[i], a, phmodel)
    end

end

@bkernel function add_photoionization_term!((tree, level, blkpos, blk), dn, u, a, phmodel)
    isleaf(tree, level, blkpos) || return
    for I in validindices(dn[1])
        for k in phmodel.target
            dn[k][I, blk] += a * u[I, blk]
        end
    end        
end
