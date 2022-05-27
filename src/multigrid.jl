#=
  Solving the Poisson equation with multigrid.
=#


"""
    Solve the equation Lu + s b = 0 where L is the 
    laplace operator (with unit grid spacing). s is the factor at level 1,
    if you are solving Poisson's equation ∇²u = -ρ/ϵ, set s = h^2/ϵ where
    h is the grid spacing at level 1.  If you want to use electron and ion
    densities, use s = eh^2/ϵ.
"""
function fmg!(u, b, r, u1, s,
              tree, conn, geometry, bc, lpldisc;
              lmax=0, ndown=3, ntop=5, nup=3, ω=1.25)
    
    lmax = length(tree)
    for l in lmax:-1:2
        sl = s / (1 << (l - 1))^2
        restrict_level!(u, conn.child[l])
        fill_ghost!(u, l - 1, conn, bc)
        
        residual_level!(r, u, b, sl, tree[l], lpldisc, geometry)
        fill_ghost!(r, l, conn, bc)
        
        restrict_level!(r, conn.child[l])
        fill_ghost!(r, l - 1, conn, bc)
        
        residual_postrestrict_level!(b, u, r, 4sl, tree[l - 1],
                                     conn.child[l], geometry, lpldisc)
        fill_ghost!(b, l - 1, conn, bc)        

        # residual_postrestrict_level!(r, u, b, 4sl, tree[l - 1],
        #                              conn.child[l], geometry, lpldisc)
        # fill_ghost!(r, l - 1, conn, bc)
    end

    for l in 1:lmax
        copyto!(u1, u, tree[l])

        if l > 1
            diffto!(u1, u, u1, tree[l - 1])
            
            fill_ghost!(u1, l - 1, conn, bc)        

            interp_add_level!(u, u1, conn.child[l])
            
            fill_ghost!(u, l - 1, conn, bc)        
        end

        vcycle!(u, b, r, u1, s, tree, conn, geometry, bc, lpldisc; lmax=l,
               ndown, ntop, nup, ω)
    end
end

"""
    Multigrid V-cycle.  Solve the equation Lu + s b = 0 where L is the 
    laplace operator (with unit grid spacing). s is the factor at level 1,
    if you are solving Poisson's equation ∇²u = -ρ/ϵ, set s = h^2/ϵ where
    h is the grid spacing at level 1.  If you want to use electron and ion
    densities, use s = eh^2/ϵ.
"""
function vcycle!(u, b, r, u1, s,
                 tree, conn, geometry, bc, lpldisc;
                 lmax=0, ndown=3, ntop=5, nup=3, ω=1.25)
    (lmax == 0) && (lmax = length(tree))
    
    # The "descending" side of the V
    for l in lmax:-1:2
        sl = s / (1 << (l - 1))^2

        gauss_seidel_iter!(u, b, ω, sl, ndown, l, tree, conn, geometry, bc, lpldisc)

        # `child` stores parent-child relations at the level of the child
        # so this restricts to level l - 1.
        restrict_level!(u, conn.child[l])
        fill_ghost!(u, l - 1, conn, bc)
        
        copyto!(u1, u, tree[l - 1])

        residual_level!(r, u, b, sl, tree[l], lpldisc, geometry)
        fill_ghost!(r, l, conn, bc)

        restrict_level!(r, conn.child[l])
        fill_ghost!(r, l - 1, conn, bc)
        
        residual_postrestrict_level!(b, u, r, 4sl, tree[l - 1],
                                     conn.child[l], geometry, lpldisc)        
        fill_ghost!(b, l - 1, conn, bc)
    end
    
    gauss_seidel_iter!(u, b, ω, s, ntop, 1, tree, conn, geometry, bc, lpldisc)
    fill_ghost!(u, 1, conn, bc)
    
    # The ascending side of the V
    for l in 2:lmax
        sl = s / (1 << (l - 1))^2

        diffto!(u1, u, u1, tree[l - 1])

        fill_ghost!(u1, l - 1, conn, bc)

        interp_add_level!(u, u1, conn.child[l])
        fill_ghost!(u, l, conn, bc)

        gauss_seidel_iter!(u, b, ω, sl, nup, l, tree, conn,
                           geometry, bc, lpldisc)

        fill_ghost!(u, l, conn, bc)
    end
end

function fill_ghost!(u, l, conn, bc)
    fill_ghost_copy!(u, conn.neighbor[l])
    fill_ghost_bnd!(u, conn.boundary[l], bc)
    fill_ghost_interp!(u, conn.refboundary[l])
end


""" 
    Compute residuals in the subblocks that have been updated with a restriction.
"""
function residual_postrestrict_level!(r, u, b, s, layer, v, geometry, lpldisc)
    @batch for edge in v
        blk = edge.coarse
        blkpos = layer.coord[blk]
        residual_subblock!(r, u, b, s, blkpos, edge.coarse,
                           geometry, lpldisc, edge.subblock)
    end
end

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
    
    hinds = CartesianIndices(ntuple(d -> d == 1 ? validrange2(u) : validrange(u),
                                    Val(D)))

    for I1 in hinds
        p = rem(P + reduce((x, y) -> rem(x + y, 2), Tuple(I1)[2:end]), 2)
        I = Base.setindex(I1, I1[1] + p, 1)

        J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
        c = diagelm(ld, geom, Val(:lhs), J)
        
        Lu = applystencil(ublk, I, J, ld, geom, Val(:lhs))
        Rb = applystencil(bblk, I, J, ld, geom, Val(:rhs))

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
    Apply GS both read and black at a given level `n` times, updating the ghost cells
    as needed.  This function can be safely iterated
"""
function gauss_seidel_iter!(u, b, ω, s, n, l, tree, conn, geometry, bc, lpldisc)
    for i in 1:n
        gauss_seidel_level!(u, b, ω, s, tree[l],
                            lpldisc, geometry, Val(0))
        
        fill_ghost_copy!(u, conn.neighbor[l])
        fill_ghost_bnd!(u, conn.boundary[l], bc)
        
        gauss_seidel_level!(u, b, ω, s, tree[l],
                            lpldisc, geometry, Val(1))
        
        fill_ghost_copy!(u, conn.neighbor[l])
        fill_ghost_bnd!(u, conn.boundary[l], bc)            
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
        
        Lu = applystencil(ublk, I, J, ld, geom, Val(:lhs))
        Rb = applystencil(bblk, I, J, ld, geom, Val(:rhs))

        rblk[I] = -(Rb + Lu / s)
    end
end


"""
    Compute the residual only in a given `subblock`.

    See residual! for details on the other parameters
"""
function residual_subblock!(r::ScalarBlockField{D, M, G},
                            u::ScalarBlockField{D, M, G},
                            b::ScalarBlockField{D, M, G}, s, blkpos, blk,
                            geom, ld::LaplacianDiscretization{D}, subblock) where {D, M, G}
    rblk = getblk(r, blk)
    ublk = getblk(u, blk)
    bblk = getblk(b, blk)
    
    gbl0 = global_first(blkpos, M)
    
    for I in subblockindices(r, subblock)
        J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
        
        Lu = applystencil(ublk, I, J, ld, geom, Val(:lhs))
        Rb = applystencil(bblk, I, J, ld, geom, Val(:rhs))

        rblk[I] = -(Rb + Lu / s)
    end
end

"""
    Compute residuals for all blocks in a layer. 
"""
function residual_level!(r, u, b, s, layer, lpldisc, geometry)
    @batch for i in eachindex(layer.pairs)
        (coord, blk) = layer.pairs[i]
        residual!(r, u, b, s, coord, blk, lpldisc, geometry)
    end
end

function resnorm!(r, b, u, s,
                  tree, conn, geometry, bc, lpldisc)
    lmax = length(tree)
    
    for l in lmax:-1:1
        sl = s / (1 << (l - 1))^2
        residual_level!(r, u, b, sl, tree[l], lpldisc, geometry)
    end

    # r .= r .^ 2
    @batch for i in eachindex(r.val)
        m = r.val[i]
        m .= m.^2
    end
    

    for l in (lmax - 1):-1:1
        restrict_level!(r, conn.child[l + 1])
    end

    w = zero(eltype(r))
    for (coord, blk) in tree[1]
        w += sum(view(getblk(r, blk), validindices(r)))
    end
    
    return sqrt(w)
end
