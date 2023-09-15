#=
Chemistry stuff, including impact ionization.
=#

abstract type AbstractChemistry end

"""
Compute derivatives of electron density and heavy species due to the chemical model `chem`.
Receives the electron density `ne` heavy species density `nh` (as an SVector) and the electric field
magnitude `eabs`, all at a given location.
"""
derivs(chem::AbstractChemistry, ne, nh, eabs) = error("Not implemented")


"""
Compute the net charge at a given location, with electron and heavy densities in `ne` and `nh`.
"""
function netcharge(chem::AbstractChemistry, n)
    return dot(n, species_charge(chem))
end


"""
Compute the chemistry-part of derivatives in a block/tree.
If `init` is Val{true}() also initializes the derivatives in the same pass.
If `prephoto` is Val{true}, computes the only the derivatives that will be used as a 
source for photo-ionization.
"""
@bkernel function chemderivs!((tree, level, blkpos, blk),
                              dn::SVector{K, <:ScalarBlockField{D, M, G}},
                              n::SVector{K, <:ScalarBlockField{D, M, G}},
                              eabs::ScalarBlockField{D, M, G},
                              chem, prephoto,
                              init::Val{vinit}=Val(true),
                              ) where {D, M, G, K, vinit}
    isleaf(tree, level, blkpos) || return

    for I in validindices(dn[1])
        dn1 = derivs(chem, species(n, I, blk), eabs[I, blk], prephoto)
        if vinit
            for i in 1:K
                dn[i][I, blk] = dn1[i]
            end
        else
            for i in 1:K
                dn[i][I, blk] += dn1[i]
            end
        end
    end
end



"""
A limited 2-species chemistry model derived from a transport model.
"""
struct NetIonization{TR <: AbstractTransportModel} <: AbstractChemistry
    trans::TR
end

# Before computing photo-ionization: we only include ionization
function derivs(chem::NetIonization, n, eabs, prephoto::Val{true})
    # n[1] is the electron density
    dne = mobility(chem.trans, eabs) * eabs * n[1] * townsend(chem.trans, eabs)
    return @SVector [dne, dne]
end

# After computing photo-ionization: everything else, including attachment
function derivs(chem::NetIonization, n, eabs, prephoto::Val{false})
    # n[1] is the electron density
    dne = -mobility(chem.trans, eabs) * eabs * n[1] * attachment(chem.trans, eabs)
    return @SVector [dne, dne]
end

@inline species_charge(::NetIonization) = @SVector([-1, 1])


"""
A limited 2-species chemistry model derived from a lookup table.
    """
@kwdef struct NetIonizationLookup{L <: LookupTable} <: AbstractChemistry
    lookup::L
    ionization_index::Int
    attachment_index::Int
end

@inline species_charge(::NetIonizationLookup) = @SVector([-1, 1])

# Before computing photo-ionization: we only include ionization
function derivs(chem::NetIonizationLookup, n, eabs, prephoto::Val{true})
    # n[1] is the electron density
    dne = n[1] * chem.lookup(eabs, chem.ionization_index)
    return @SVector [dne, dne]
end

# After computing photo-ionization: everything else, including attachment
function derivs(chem::NetIonizationLookup, n, eabs, prephoto::Val{false})
    # n[1] is the electron density
    dne = n[1] * chem.lookup(eabs, chem.attachment_index)
    return @SVector [dne, dne]
end


