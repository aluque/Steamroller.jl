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
function netcharge(chem::AbstractChemistry, ne, nh)
    return dot(nh, species_charge(chem)) - ne
end


"""
Compute the chemistry-part of derivatives in a block/tree.
If `init` is Val{true}() also initializes the derivatives in the same pass.
"""
@bkernel function chemderivs!((tree, level, blkpos, blk),
                              dne::ScalarBlockField{D, M, G},
                              dnh::ScalarBlockField{D, M, G},                              
                              ne::ScalarBlockField{D, M, G},
                              nh::ScalarBlockField{D, M, G},
                              eabs::ScalarBlockField{D, M, G},
                              chem, init::Val{vinit}=Val(true)) where {D, M, G, vinit}
    for I in validindices(dne)
        dne1, dnh1 = derivs(chem, ne[I, blk], nh[I, blk], eabs[I, blk])        
        if vinit
            dne[I, blk] = dne1
            dnh[I, blk] = dnh1
        else
            dne[I, blk] += dne1
            dnh[I, blk] += dnh1
        end
    end
end



"""
A limited chemistry model derived from a transport model.
"""
struct NetIonization{TR <: AbstractTransportModel} <: AbstractChemistry
    trans::TR
end

function derivs(chem::NetIonization, ne, nh, eabs)
    dne = mobility(chem.trans, eabs) * eabs * ne * nettownsend(chem.trans, eabs)
    dh = @SVector [dne]

    return (dne, dh)
end

@inline species_charge(::NetIonization) = @SVector([1])
