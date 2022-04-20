#=
  Solving the Poisson equation with multigrid.
=#

@generated function residual!(r::ScalarBlockField{D, M, G},
                              u::ScalarBlockField{D, M, G},
                              b::ScalarBlockField{D, M, G}, s, blkpos, blk,
                              stencil_type::ST,
                              geometry_type::GT) where {D, M, G, ST, GT}

    # Expands to something like :(ublk[i_1, i_2]); ublk and bblk are defined
    # inside the generated function
    uref = macroexpand(@__MODULE__, :(@nref $D ublk i))
    bref = macroexpand(@__MODULE__, :(@nref $D bblk i))

    st = ST()
    gt = GT()

    # Global coordinates.
    glcoords = macroexpand(@__MODULE__, :(@ntuple $D j))
    
    # Expands to the application of the stencil around e.g. u[i_1, i_2]
    lhs = stencilexpr(lstencil(st), uref, glcoords.args, gt)
    rhs = stencilexpr(rstencil(st), bref, glcoords.args, gt)

    quote
        rblk = getblk(r, blk)
        ublk = getblk(u, blk)
        bblk = getblk(b, blk)

        gbl0 = global_first(blkpos, $M)
        
        @nloops $D i d->validrange(r) begin
            @nexprs $D d->(j_d = i_d - gbl0[d] - $G)
            (@nref $D rblk i) = -(($rhs) + ($lhs) / s)
        end
    end
end


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
