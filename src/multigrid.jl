#=
  Solving the Poisson equation with multigrid.
=#


"""
    Compute the residual of applying the laplace operator to a potential
    field `u` with source field `b` and store the result in `r`.

    The function computes -(Lu/s + b) where L is the discrete laplace operator.
    `blk` is the block index and `blkpos` is a `CartesianIndex` with the
    coordinates of `blk` in the block mesh.  Needs a stencil passed as a type 
    in `stencil_type` and a geometry also as a type in `geometry_type`.
    `s` is a constant scalar factor
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
        J = CartesianIndex(ntuple(d -> I[d] - G - gbl0[d], Val(D)))
        
        Lu = applystencil(ublk, I, J, geom, ld, Val(:lhs))
        Rb = applystencil(bblk, I, J, geom, ld, Val(:rhs))

        rblk[I] = - (Rb + Lu / s)
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
