#=
  Solving the Poisson equation with multigrid.
=#


"""
    Compute the parity of a CartesianIndex.
"""
@inline function parity(I::NTuple{D}) where D
    p = rem(I[1], 2)
    for i in 2:D
        p = xor(p, rem(I[i], 2))
    end

    return p
end

@inline parity(I::CartesianIndex) = parity(Tuple(I))

"""
    Apply Red-black Gauss-Seidel in a block.

    # Parameters 
    * `u` is the (to be updated) solution guess.
    * `b` is the source term.
    * `ω` is an over-relaxation parameter.
    * `s` is a scaling parameter.
    * `blkpos` is the block coordinate of the block.
    * `blk` is the block index.
    * `ld` is a `LaplacianDiscretization`
    * `geom` is a geometry.
    * `parity` should be either Val(0) or Val(1).
"""
function gauss_seidel!(u::ScalarBlockField{D, M, G},
                       b::ScalarBlockField{D, M, G}, ω, s, blkpos, blk,
                       ld::LaplacianDiscretization{D},
                       geom::GT, par::Val{P}) where {D, M, G, GT, P}
    ublk = getblk(u, blk)
    bblk = getblk(b, blk)
    
    gbl0 = global_first(blkpos, M)
    c = center_factor(ld, Val(:lhs), geom) / denom(ld, Val(:lhs), geom)
    
    hinds = CartesianIndices(ntuple(d -> d == 1 ? validrange2(u) : validrange(u),
                                    Val(D)))

    for I1 in hinds
        p = rem(P + reduce((x, y) -> rem(x + y, 2), Tuple(I1)[2:end]), 2)
        I = Base.setindex(I1, I1[1] + p, 1)

        J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
        
        Lu = applystencil(ublk, I, J, geom, ld, Val(:lhs))
        Rb = applystencil(bblk, I, J, CartesianGeometry(), ld, Val(:rhs))

        ublk[I] -= ω * (Lu + s * Rb) / c
    end
end
    

"""
    Apply Gauss-Seidel to all blocks in a layer. 
"""
function gauss_seidel_level!(u, b, ω, s, layer, ld, geometry, parity)
    @batch for i in eachindex(layer.pairs)
        (coord, blk) = layer.pairs[i]
        gauss_seidel!(u, b, ω, s, coord, blk, ld, geometry, parity)
    end
end


"""
    Compute the residual of applying the laplace operator to a potential
    field `u` with source field `b` and store the result in `r`.

    The function computes -(Lu/s + b) where L is the discrete laplace operator.
    `blk` is the block index and `blkpos` is a `CartesianIndex` with the
    coordinates of `blk` in the block mesh.  Needs a stencil passed as a type 
    in `stencil_type` and a geometry also as a type in `geom`.
    `s` is a constant scalar factor.
"""
function residual!(r::ScalarBlockField{D, M, G},
                   u::ScalarBlockField{D, M, G},
                   b::ScalarBlockField{D, M, G}, s, blkpos, blk,
                   ld::LaplacianDiscretization{D},
                   geom::GT) where {D, M, G, GT}
    rblk = getblk(r, blk)
    ublk = getblk(u, blk)
    bblk = getblk(b, blk)
    
    gbl0 = global_first(blkpos, M)
    
    for I in CartesianIndices(ntuple(d -> validrange(r), Val(D)))
        J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
        
        Lu = applystencil(ublk, I, J, geom, ld, Val(:lhs))
        Rb = applystencil(bblk, I, J, geom, ld, Val(:rhs))

        rblk[I] = -(Rb + Lu / s)
    end
end


"""
    Compute residuals for all blocks in a layer. 
"""
function residual_level!(r, u, b, s, layer, stencil, geometry)
    @batch for i in eachindex(layer.pairs)
        (coord, blk) = layer.pairs[i]
        residual!(r, u, b, s, coord, blk, stencil, geometry)
    end
end
