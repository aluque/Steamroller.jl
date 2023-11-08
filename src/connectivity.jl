#= 
Handling of connectivity: the spatial relationships between blocks and between blocks and boundaries.
=#

# The block is at a domain boundary
struct Boundary{D}
    block::BlockIndex
    face::CartesianIndex{D}
    bnd::CartesianIndex{D}
    level::Int
end

# The block from and to share a face at the same level
struct Neighbor{D}
    from::BlockIndex
    to::BlockIndex

    # face refers to the face in `from`
    face::CartesianIndex{D}
end

# The block fine is at a refinement boundary inside / facing coarse
struct RefBoundary{D}
    fine::BlockIndex
    coarse::BlockIndex

    # face refers to the face in `fine`
    face::CartesianIndex{D}

    # the subblock in coarse where the interface will sit
    subblock::CartesianIndex{D}
end


# The block fine is below the block coarse
struct Child{D}
    fine::BlockIndex
    coarse::BlockIndex

    # the subblock in coarse that covers fine
    subblock::CartesianIndex{D}
end



""" 
Struct to contain the relationship between blocks in a given tree
structure.  We use lists for each level to denote relationships.
$(FIELDS)
"""
struct Connectivity{D}    
    "For each level, list of blocks adjacent to a boundary."
    boundary::Vector{Vector{Boundary{D}}}

    "For each level, list of neighbor relationships."
    neighbor::Vector{Vector{Neighbor{D}}}

    "For each level, list of refinement boundaries."
    refboundary::Vector{Vector{RefBoundary{D}}}

    """For each level, list of parents-children relationship
    (stored at the level of the children)."""
    child::Vector{Vector{Child{D}}}
end

function Connectivity{D}(levels) where {D}
    Connectivity(map(l -> Boundary{D}[], 1:levels),
                 map(l -> Neighbor{D}[], 1:levels),
                 map(l -> RefBoundary{D}[], 1:levels),
                 map(l -> Child{D}[], 1:levels))
end

__connections(c::Connectivity, ::Type{Boundary{D}}, level) where {D} = c.boundary[level]
__connections(c::Connectivity, ::Type{Neighbor{D}}, level) where {D} = c.neighbor[level]
__connections(c::Connectivity, ::Type{RefBoundary{D}}, level) where {D} = c.refboundary[level]
__connections(c::Connectivity, ::Type{Child{D}}, level) where {D} = c.child[level]

Base.push!(c::Connectivity, level, item::T) where {T} = push!(__connections(c, T, level), item)
function Base.empty!(c::Connectivity)
    for l in eachindex(c.boundary)
        empty!(c.boundary[l])
        empty!(c.neighbor[l])
        empty!(c.refboundary[l])
        empty!(c.child[l])
    end
end


""" 
Find connectivities in `tree` represented by a list of block layers.
Neighbors are found according to `stencil` and the behaviour is defined
by a `handler` object that should have defined methods `boundary`, 
`neighbor` and `parent`.
"""
function connectivity(tree::Tree{D}, stencil) where {D}
    levels = length(tree)
    conn = Connectivity{D}(levels)
    connectivity!(conn, tree, stencil)
    return conn
end


"""
Same as `connectivity`, but updates a previously created instance passed as `conn`.
"""
function connectivity!(conn, tree::Tree{D}, stencil) where {D}
    empty!(conn)
    for layer in tree
        for (c, blk) in layer
            for s in stencil
                if !((c + s) in layer.domain)
                    # For box stencils we must find the proper boundary
                    bnd = CartesianIndex(ntuple(d -> (c[d] + s[d]) in layer.domain.indices[d] ?
                                                0 : s[d], Val(D)))
                    # Block at the domain boundary
                    push!(conn, layer.level, Boundary(blk, s, bnd, layer.level))
                    continue
                end

                other = get(layer, c + s)
                if other != 0
                    # A neighbor exists
                    push!(conn, layer.level, Neighbor(blk, other, s))
                elseif layer.level > 1
                    # A refinement boundary
                    pc = parentcoord(c + s)
                    pblk = get(tree[layer.level - 1], pc)

                    @assert pblk != 0 "Proper nesting broken" 

                    push!(conn, layer.level,
                          RefBoundary(blk, pblk, s, subcoord(c + s)))
                end
            end

            # Parent coordinate
            if layer.level > 1
                pc = parentcoord(c)
                pblk = get(tree[layer.level - 1], pc)

                @assert pblk != 0 "Malformed tree"
                
                push!(conn, layer.level, Child(blk, pblk, subcoord(c)))
            end
        end
    end

    return conn
end


function fill_ghost!(u, l, conn, bc)
    fill_ghost_copy!(u, conn.neighbor[l])
    fill_ghost_bnd!(u, conn.boundary[l], bc)
    fill_ghost_interp!(u, conn.refboundary[l])
end


function fill_ghost_conserv!(u, l, conn, bc)
    fill_ghost_copy!(u, conn.neighbor[l])
    fill_ghost_bnd!(u, conn.boundary[l], bc)
    fill_ghost_conserv!(u, conn.refboundary[l])
end


"""
Fill ghost cells by copying neighboring cells in the same layer.
`v` contains a vector with `Neighbor` relations.
"""
function fill_ghost_copy!(u::ScalarBlockField{D, M, G}, v::Vector{Neighbor{D}}) where {D, M, G}
    @batch for edge in v
        @inline _fill_ghost_copy!(u, edge)
    end
end

@generated function _fill_ghost_copy!(u::ScalarBlockField{D, M, G}, edge::Neighbor{D}) where {D, M, G}
    quote
        f = edge.face
        
        ufrom = u[edge.from]
        uto = u[edge.to]
        
        @nloops $D i d->(f[d] == 0 ? (1:M) : (1:G)) begin
            ifrom = @ntuple $D d->(f[d] == 0 ? G + i_d : (f[d] == -1 ? i_d + G : i_d + M))
            ito =   @ntuple $D d->(f[d] == 0 ? G + i_d : (f[d] == 1  ? i_d : i_d + M + G))
            
            uto[ito...] = ufrom[ifrom...]
        end
    end
end


"""
Copy ghost cells from neighbouring blocks according to a full connectivity
structure `conn`.
"""
function fill_ghost_copy!(u::ScalarBlockField, conn::Connectivity)
    # For each layer...
    for v in conn.neighbor
        fill_ghost_copy!(u, v)
    end
end


"""
Fill ghost cells in the boundary of a given layer by applying the 
boundary conditions specified by `bc`
"""
function fill_ghost_bnd!(u::ScalarBlockField{D}, v::Vector{Boundary{D}},
                         bc::HomogeneousBoundaryConditions) where {D}
    @batch for link in v
        ghost = ghostindices(u, link.face)
        valid = mirrorghost(u, ghost, link.bnd)
        
        ublk = u[link.block]
        CI = CartesianIndices(ghost)
        
        for I in CI
            # I1 = __mirrorindex(I, CI, link.bnd)
            ublk[ghost[I]] = getbc(bc, link.bnd, ublk[valid[I]])
        end
    end
end


"""
Fill ghost cells in boundary blocks according to given boundary conditions 
`bc`.
structure `conn`.
"""
function fill_ghost_bnd!(u::ScalarBlockField, conn::Connectivity, bc)
    # For each layer...
    for v in conn.boundary
        fill_ghost_bnd!(u, v, bc)
    end
end


"""
Fill ghost cells by interpolating from parent blocks at refinement 
boundaries.
"""
function fill_ghost_interp!(u::ScalarBlockField{D}, v::Vector{RefBoundary{D}}, method=InterpLinear()) where {D}
    @batch for edge in v
        src = u[edge.coarse]
        dest = u[edge.fine]
        interp!(method, dest, ghostindices(u, edge.face),
                src, subblockbnd(u, edge.subblock, -edge.face))
    end
end

"""
Fill ghost cells by interpolating from parent blocks at all refinement 
boundaries.
"""
function fill_ghost_interp!(u::ScalarBlockField{D}, conn::Connectivity) where {D}
    # For each layer...
    for v in conn.refboundary
        fill_ghost_interp!(u, v)
    end
end


"""
Fill ghost cells according to the conservative method described by Teunissen 2017.

`fine` is the fine block, `coarse` is the coarse block, `ifine` are `CartesianIndices` into the ghost cells
to be filled in `fine`, `face` is the direction from `fine` to `coarse`.
"""
@generated function fill_ghost_conserv!(fine::AbstractArray{T, D}, ifine::CartesianIndices{D},
                                        coarse::AbstractArray{T, D}, icoarse::CartesianIndices{D},
                                        face::CartesianIndex{D}) where {T, D}
        quote
            # If, Ic -> Indexes of fine, cooarse inside the rectangles
            # Jf, Jc -> Indexes in the original arrays (e.g. Jf = ifine[Is])
            for If in CartesianIndices(ifine)
                Ic = CartesianIndex(@ntuple $D d -> fld1(If[d], 2))
                Jc = icoarse[Ic]
                Jf = ifine[If]
                
                f = __conserv_coeff0(Val($D)) * coarse[Jc]
                Jf1 = CartesianIndex(@ntuple $D d->(Jf[d] - face[d]))
                f += __conserv_coeff1(Val($D)) * fine[Jf1]
                for d1 in 1:$D
                    if face[d1] != 0
                        Jf2 = CartesianIndex(@ntuple $D d->(d==d1 ? Jf1[d] - face[d] : Jf1[d]))
                    else
                        sg = 2 * (2 - mod1(Jf1[d1], 2)) - 1
                        Jf2 = CartesianIndex(@ntuple $D d->(d==d1 ? Jf1[d] + sg : Jf1[d]))
                    end
                    f += __conserv_coeff2(Val($D)) * fine[Jf2]
                end                
                fine[Jf] = f
            end
        end
end

function fill_ghost_conserv!(u::ScalarBlockField{D}, v::Vector{RefBoundary{D}}) where {D}
    @batch for edge in v
        src = u[edge.coarse]
        dest = u[edge.fine]
        fill_ghost_conserv!(dest, ghostindices(u, edge.face),
                            src, subblockbnd(u, edge.subblock, -edge.face), edge.face)
    end
end


function fill_ghost_conserv!(u::ScalarBlockField{D}, conn::Connectivity) where {D}
    # For each layer...
    for v in conn.refboundary
        fill_ghost_conserv!(u, v)
    end
end

"""
Fill the flux of a coarse block restricting from the flux of a finer lock in a refinement
boundary.
"""
function restrict_flux!(f::VectorBlockField{D}, v::Vector{RefBoundary{D}}) where {D}
    for edge in v
        (;coarse, fine, face, subblock) = edge        

        # We do not consider fluxes in the diagonal direction.  Generally this is not a problem
        # but if the connectivity was built using a BoxStencil, we also have diagonal refinement
        # boundaries.  This check avoids problems in that case.
        isaxis(face) || continue
        
        isrc = bndface(f, face)
        idest1 = bndface(f, -face)
        dim = perpdim(face)
        #@info "Restricting flux" coarse fine face dim
        
        idest = CartesianIndices(ntuple(d -> d == dim ? idest1.indices[d] :
                                        halfrange(idest1.indices[d], subblock[d]),
                                        Val(D)))
        
        restrict_flux!(f[coarse], idest, f[fine], isrc, dim)
    end
end


"""
Fill the flux of a coarse block restricting from the flux of a finer lock in a refinement
boundary.
"""
function restrict_flux!(f::VectorBlockField{D}, conn::Connectivity) where {D}
    # For each layer...
    for (lvl, v) in enumerate(conn.refboundary)
        #@info "Restricting flux at level" lvl
        restrict_flux!(f, v)
    end
end


@inline function __mirrorindex(I::CartesianIndex{D}, CI::CartesianIndices{D},
                        bnd::CartesianIndex{D}) where {D}
    CartesianIndex(ntuple(d -> (bnd[d] == 0 ? I[d] :
                                (last(CI.indices[d]) - I[d] + 1)), Val(D)))
end

