#=
  Solving the Poisson equation with multigrid.
=#


"""
Solve the equation Lu + s b = 0 where L is the 
laplace operator (with unit grid spacing). s is the factor at level 1,
if you are solving Poisson's equation ∇²u = -ρ/ϵ, set s = h^2/ϵ where
h is the grid spacing at level 1.  If you want to use electron and ion
densities, use s = eh^2/ϵ.

With a discretization of type `HelmholtzDiscretization` this can also be used to solve
Helmholtz's equation in the form Lu - s k² u + s b = 0.  Usually you want s = h^2 with h
the grid spacing at level 1.

`ghost` selects the interpolation used to fill ghost cells at refinement boundaries.
"""
function fmg!(u, b, r, u1, s,
              tree, conn, geometry, bc, lpldisc;
              lmax=0, ndown=3, ntop=5, nup=3, ω=1.25, ghost=InterpLinear())

    lmax = findlast(!isempty, tree)
    for l in lmax:-1:2
        sl = s / (1 << (l - 1))^2
        restrict_level!(u, conn.child[l])
        fill_ghost!(u, l - 1, conn, bc, ghost)

        # The ghosts of r are never read: restrict! reads only valid cells and
        # residual_postrestrict_level! reads r only at valid cells.
        residual_level!(r, u, b, sl, tree[l], lpldisc, geometry)
        restrict_level!(r, conn.child[l])

        residual_postrestrict_level!(b, u, r, 4sl, tree[l - 1],
                                     conn.child[l], geometry, lpldisc)
    end

    for l in 1:lmax
        copyto!(u1, u, tree[l])

        if l > 1
            diffto!(u1, u, u1, tree[l - 1])
            fill_ghost!(u1, l - 1, conn, bc, ghost)
            interp_add_level!(u, u1, conn.child[l])
        end

        fill_ghost!(u, l, conn, bc, ghost)

        vcycle!(u, b, r, u1, s, tree, conn, geometry, bc, lpldisc; lmax=l,
               ndown, ntop, nup, ω, ghost)
    end
end

"""
Multigrid V-cycle.  Solve the equation Lu + s b = 0 where L is the 
laplace operator (with unit grid spacing). s is the factor at level 1,
if you are solving Poisson's equation ∇²u = -ρ/ϵ, set s = h^2/ϵ where
h is the grid spacing at level 1.  If you want to use electron and ion
densities, use s = eh^2/ϵ.

`ghost` selects the interpolation used to fill ghost cells at refinement boundaries.
"""
function vcycle!(u, b, r, u1, s,
                 tree, conn, geometry, bc, lpldisc;
                 lmax=0, ndown=3, ntop=5, nup=3,
                 ω=1.0, ghost=InterpLinear())
    if lmax == 0
        lmax = finestlevel(tree)::Int

        #=
        With a BoxStencil the boundary fill of a diagonal face reads refinement-boundary ghosts,
        which fill_ghost! writes after it, so a single fill can leave corner ghosts stale (or
        uninitialized in new blocks).  The earlier default lmax = length(tree) refilled the
        finest level through the empty level lmax + 1; this keeps that second fill.
        =#
        fill_ghost!(u, lmax, conn, bc, ghost)
    end

    # The "descending" side of the V
    for l in lmax:-1:2
        sl = s / (1 << (l - 1))^2

        gauss_seidel_iter!(u, b, ω, sl, ndown, l, tree, conn, geometry, bc, lpldisc)

        # `child` stores parent-child relations at the level of the child
        # so this restricts to level l - 1.
        restrict_level!(u, conn.child[l])
        fill_ghost!(u, l - 1, conn, bc, ghost)

        copyto!(u1, u, tree[l - 1])

        # The ghosts of r are never read: restrict! reads only valid cells and
        # residual_postrestrict_level! reads r only at valid cells.
        residual_level!(r, u, b, sl, tree[l], lpldisc, geometry)
        restrict_level!(r, conn.child[l])

        residual_postrestrict_level!(b, u, r, 4sl, tree[l - 1],
                                     conn.child[l], geometry, lpldisc)
    end

    gauss_seidel_iter!(u, b, ω, s, ntop, 1, tree, conn, geometry, bc, lpldisc)

    # The ascending side of the V
    for l in 2:lmax
        sl = s / (1 << (l - 1))^2

        diffto!(u1, u, u1, tree[l - 1])

        fill_ghost!(u1, l - 1, conn, bc, ghost)

        interp_add_level!(u, u1, conn.child[l])
        fill_ghost!(u, l, conn, bc, ghost)

        # This leaves all ghosts of u at level l up to date: gauss_seidel_iter! ends with copy
        # and boundary fills and the refinement-boundary ghosts depend only on level l - 1.
        gauss_seidel_iter!(u, b, ω, sl, nup, l, tree, conn, geometry, bc, lpldisc)
    end
end


""" 
Compute residuals in the subblocks that have been updated with a restriction.
"""
function residual_postrestrict_level!(r, u, b, s, layer, v, geometry, lpldisc)
    @pbatch for edge in v
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

struct RedBlack{P}; end
struct FourColors{P1, P2}; end


"""
Apply Red-black Gauss-Seidel in a block.

# Parameters 
* `u` is the (to be updated) solution guess.
* `b` is the source term.
* `ω` is an over-relaxation parameter.
* `s` is a scaling parameter.
* `blkpos` is the block coordinate of the block.
* `blk` is the block index.
* `ld` is a `AbstractDiscretization`
* `geom` is a geometry.
* `parity` should be either Val(0) or Val(1).
"""
@generated function gauss_seidel!(u::ScalarBlockField{D, M, G},
                                   b::ScalarBlockField{D, M, G}, ω, s, blkpos, blk,
                                   ld::AbstractDiscretization{D},
                                   geom::GT, par::RedBlack{P}) where {D, M, G, GT, P}
    D1 = D - 1
    quote
        gbl0 = global_first(blkpos, $M)
        checkblock(geom, gbl0, $M)
        ublk = u[blk]
        bblk = b[blk]

        # Index 1 steps in strides of 2; all other indices have strides of one
        @inbounds @nloops $D i d->(d == 1 ? ((G + 1):2:(G + M)) : ((G + 1):(G + M))) begin
            # Compute parity of P combined with indices 2, 3...
            p = $P
            @nexprs $D1 d->(p = xor(p, i_{d+1} & 1))
            
            # Local index
            I = CartesianIndex(@ntuple $D d->(d == 1 ? i_d + p : i_d))            

            # Global index.  This is needed for cylindrical geometry.
            J = CartesianIndex(@ntuple $D d->(I[d] - G + gbl0[d] - 1))

            c = diagelm(s, ld, geom, Val(:lhs), J)            
            Lu = applystencil(ublk, s, I, J, ld, geom, Val(:lhs))
            
            #u[I, blk] -= ω * (Lu + s * b[I, blk]) / c
            v = muladd(s, bblk[I], Lu)
            ublk[I] = muladd(-(ω / c), v, ublk[I])
        end
    end
end


# Non-@generated version, kept for reference
function _gauss_seidel!(u::ScalarBlockField{D, M, G},
                       b::ScalarBlockField{D, M, G}, ω, s, blkpos, blk,
                       ld::AbstractDiscretization{D},
                       geom::GT, par::RedBlack{P}) where {D, M, G, GT, P}
    gbl0 = global_first(blkpos, M)
    
    hinds = CartesianIndices(ntuple(d -> d == 1 ? validrange2(u) : validrange(u),
                                    Val(D)))

    for I1 in hinds
        p = rem(P + reduce((x, y) -> rem(x + y, 2), Tuple(I1)[2:end]), 2)
        I = Base.setindex(I1, I1[1] + p, 1)

        J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
        c = diagelm(s, ld, geom, Val(:lhs), J)
        
        Lu = applystencil(u[blk], s, I, J, ld, geom, Val(:lhs))

        u[I, blk] -= ω * (Lu + s * b[I, blk]) / c
    end
end

function gauss_seidel!(u::ScalarBlockField{D, M, G},
                       b::ScalarBlockField{D, M, G}, ω, s, blkpos, blk,
                       ld::AbstractDiscretization{D},
                       geom::GT, par::FourColors{P1, P2}) where {D, M, G, GT, P1, P2}
    gbl0 = global_first(blkpos, M)
    
    hinds = CartesianIndices(ntuple(d -> d <= 2 ? validrange2(u) : validrange(u),
                                    Val(D)))
    checkblock(geom, gbl0, M)
    ublk = u[blk]
    bblk = b[blk]

    @inbounds for I1 in hinds
        p1 = P1
        p2 = P2
        if D == 3
            p1 = rem(p1 + I1[3], 2)
            p2 = rem(p2 + I1[3], 2)
        end
        
        I = Base.setindex(Base.setindex(I1, I1[1] + p1, 1), I1[2] + p2, 2)

        J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
        c = diagelm(s, ld, geom, Val(:lhs), J)
        
        Lu = applystencil(ublk, s, I, J, ld, geom, Val(:lhs))

        ublk[I] -= ω * (Lu + s * bblk[I]) / c
    end
end


"""
Apply Gauss-Seidel to all blocks in a layer. 
"""
function gauss_seidel_level!(u::ScalarBlockField{D, M, G, T}, b, ω, s, layer, ld, geometry,
                             parity) where {D, M, G, T}
    # A Float64 ω (the default) would promote every update to Float64.
    ω = convert(T, ω)
    @pbatch for i in eachindex(layer.pairs)
        (coord, blk) = layer.pairs[i]
        gauss_seidel!(u, b, ω, s, coord, blk, ld, geometry, parity)
    end
end


"""
Apply GS both read and black at a given level `n` times, updating the ghost cells
as needed.  This function can be safely iterated
"""
function gauss_seidel_iter!(u, b, ω, s, n, l, tree, conn, geometry, bc,
                            lpldisc::AbstractDiscretization{D, 2}) where D
    for i in 1:n
        gauss_seidel_level!(u, b, ω, s, tree[l],
                            lpldisc, geometry, RedBlack{0}())
        
        fill_ghost_copy!(u, conn.neighbor[l])
        fill_ghost_bnd!(u, conn.boundary[l], bc)
        
        gauss_seidel_level!(u, b, ω, s, tree[l],
                            lpldisc, geometry, RedBlack{1}())
        
        fill_ghost_copy!(u, conn.neighbor[l])
        fill_ghost_bnd!(u, conn.boundary[l], bc)
    end
end

# For the 4th order discretization, use 4 colors.
function gauss_seidel_iter!(u, b, ω, s, n, l, tree, conn, geometry, bc,
                            lpldisc::AbstractDiscretization{D, 4}) where D
    for i in 1:n
        gauss_seidel_level!(u, b, ω, s, tree[l],
                            lpldisc, geometry, FourColors{0, 0}())
        
        fill_ghost_copy!(u, conn.neighbor[l])
        fill_ghost_bnd!(u, conn.boundary[l], bc)
        
        gauss_seidel_level!(u, b, ω, s, tree[l],
                            lpldisc, geometry, FourColors{0, 1}())
        
        fill_ghost_copy!(u, conn.neighbor[l])
        fill_ghost_bnd!(u, conn.boundary[l], bc)

        gauss_seidel_level!(u, b, ω, s, tree[l],
                            lpldisc, geometry, FourColors{1, 0}())
        
        fill_ghost_copy!(u, conn.neighbor[l])
        fill_ghost_bnd!(u, conn.boundary[l], bc)

        gauss_seidel_level!(u, b, ω, s, tree[l],
                            lpldisc, geometry, FourColors{1, 1}())
        
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
                   ld::AbstractDiscretization{D},
                   geom::GT) where {D, M, G, GT}
    gbl0 = global_first(blkpos, M)
    checkblock(geom, gbl0, M)
    rblk, ublk, bblk = r[blk], u[blk], b[blk]
    
    @inbounds for I in CartesianIndices(ntuple(d -> validrange(r), Val(D)))
        J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
        
        Lu = applystencil(ublk, s, I, J, ld, geom, Val(:lhs))

        rblk[I] = -(bblk[I] + Lu / s)
    end
end


"""
Compute the residual only in a given `subblock`.

See residual! for details on the other parameters
"""
function residual_subblock!(r::ScalarBlockField{D, M, G},
                            u::ScalarBlockField{D, M, G},
                            b::ScalarBlockField{D, M, G}, s, blkpos, blk,
                            geom, ld::AbstractDiscretization{D}, subblock) where {D, M, G}
    gbl0 = global_first(blkpos, M)
    checkblock(geom, gbl0, M)
    rblk, ublk, bblk = r[blk], u[blk], b[blk]
    
    @inbounds for I in subblockindices(r, subblock)
        J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
        
        Lu = applystencil(ublk, s, I, J, ld, geom, Val(:lhs))

        rblk[I] = -(bblk[I] + Lu / s)
    end
end

"""
Compute residuals for all blocks in a layer. 
"""
function residual_level!(r, u, b, s, layer, lpldisc, geometry)
    @pbatch for i in eachindex(layer.pairs)
        (coord, blk) = layer.pairs[i]
        residual!(r, u, b, s, coord, blk, lpldisc, geometry)
    end
end


"""
Compute the rhs of the Laplacian discretization.

This is only neccesary for 4th-order compact discretization where we solve
Lu + Rb = 0 (with L a lhs and R an rhs operator).
"""
function rhs!(b1::ScalarBlockField{D, M, G},
              b::ScalarBlockField{D, M, G},
              blkpos, blk,
              ld::AbstractDiscretization{D},
              geom::GT) where {D, M, G, GT}
    gbl0 = global_first(blkpos, M)
    checkblock(geom, gbl0, M)
    b1blk, bblk = b1[blk], b[blk]
    
    @inbounds for I in CartesianIndices(ntuple(d -> validrange(b1), Val(D)))
        J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
        
        # s = nothing here bc it is ignored
        Rb = applystencil(bblk, nothing, I, J, ld, geom, Val(:rhs))
        b1blk[I] = Rb
    end
end


"""
Compute rhs for all blocks in a layer. 
"""
function rhs_level!(b1, b, layer, ld, geometry)
    @pbatch for i in eachindex(layer.pairs)
        (coord, blk) = layer.pairs[i]
        rhs!(b1, b, coord, blk, ld, geometry)
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
    @batch for i in eachindex(r.u)
        m = r.u[i]
        m .= m.^2
    end
    

    for l in (lmax - 1):-1:1        
        restrict_level!(r, conn.child[l + 1])
    end

    w = zero(eltype(r))
    for (coord, blk) in tree[1]
        w += sum(view(r[blk], validindices(r)))
    end
    
    return sqrt(w)
end


""" Compute the electric field in a block in `f` and its absolute value in `fabs`. """
@bkernel function electric_field!((tree, level, coord, blk),
                                  f::VectorBlockField{D, M, G},
                                  fabs::ScalarBlockField{D, M, G},
                                  u::ScalarBlockField{D, M, G}, h,
                                  t, efunc::F) where {D, M, G, F <: Function}
    gbl0 = global_first(coord, M)
    h /= (1 << (level - 1))
    
    @inbounds for d in 1:D
        for I in validindices(f, d)
            x = SVector(ntuple(d1 -> h * (I[d1] - G + gbl0[d1] - 1) - ((d == d1) ? h : h/2), Val(D)))
            e0 = efunc(x, t)
            I1 = Base.setindex(I, I[d] - 1, d)            
            f[I, d, blk] = (u[I1, blk] - u[I, blk]) / h + e0[d]
        end
    end
    
    @inbounds for I in validindices(fabs)
        f2 = zero(eltype(fabs))
        for d in 1:D
            I1 = Base.setindex(I, I[d] + 1, d)

            f2 += (f[I, d, blk] + f[I1, d, blk])^2 / 4
        end
        fabs[I, blk] = sqrt(f2)
    end
end

@bkernel electric_field!((tree, level, coord, blk), f, fabs, u, h, t, e0::SVector) = electric_field!((tree, level, coord, blk), f, fabs, u, h, t, (x, t) -> e0)
