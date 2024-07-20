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
        for l in 1:length(tree)
            fill_ghost!(phfields[i], l, conn, bc)
        end

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


###
#  Predefined photo-ionization models
##
"""
Construct an empty photo-ionization model.  Because the number of terms (0) is known statically,
this should have a zero overhead.
"""
function empty_photoionization(T)    
    return PhotoionizationModel{0, T, Nothing}(SVector{0, PhotoionizationTerm{T}}(), 0, (0, 0), nothing, 0)    
end


"""
Construct the bourdon 2/3-term model as described by Bagheri 2018.
"""
function bourdon(T, A, λ, D=2;
                 # Index of the source (the species that contains ionization used as a source).
                 source=2,
                 # ξB νu/νi
                 ξ´ = 0.075,
                 pO2 = 150 * co.torr,
                 p = 750 * co.torr,
                 pq = 40 * co.milli * co.bar)

    # See A.9 for the multiplicative factors.  Note also the unit conversion.
    a = @. A * pO2^2 * (pq / (p + pq)) * ξ´ * co.centi^-2 * co.torr^-2
    k = @. pO2 * λ * co.centi^-1 * co.torr^-1

    term = PhotoionizationTerm{T}.(a, k .^ 2)
    bc = (D == 2 ?
        boundaryconditions(((1, 1), (-1, -1))) :
        boundaryconditions(((-1, -1), (-1, -1), (-1, -1))))
    
    phmodel = PhotoionizationModel{length(A), T, typeof(bc)}(term, 1, (1, 2), bc, 2)

    return phmodel
end

# Direct copy of table A2 in Bagheri 2018
bourdon2(T=Float64, D=2) = bourdon(T,
                                   [0.0021, 0.1775], # A
                                   [0.0974, 0.5877], # λ
                                   D)

# Direct copy of table A3 in Bagheri 2018
bourdon3(T=Float64, D=2) = bourdon(T,
                                   [1.986e-4, 0.0051, 0.4886],  # A
                                   [0.0553, 0.1460, 0.8900],    # λ
                                   D)
