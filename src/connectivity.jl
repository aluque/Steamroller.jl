#= 
Handling of connectivity: the spatial relationships between blocks and between blocks and boundaries.
=#

# The block is at a domain boundary
struct Boundary{D}
    block::BlockIndex
    face::CartesianIndex{D}
    bnd::CartesianIndex{D}
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


""" 
    Find connectivities in `tree` represented by a list of block layers.
    Neighbors are found according to `stencil` and the behaviour is defined
    by a `handler` object that should have defined methods `boundary`, 
    `neighbor` and `parent`.
"""
function connectivity(tree::Tree{D}, stencil) where {D}
    levels = length(tree)
    conn = Connectivity{D}(levels)
    
    for layer in tree
        for (c, blk) in layer
            for s in stencil
                if !((c + s) in layer.domain)
                    # For box stencils we must find the proper boundary
                    bnd = CartesianIndex(ntuple(d -> (c[d] + s[d]) in layer.domain.indices[d] ?
                                                0 : s[d], Val(D)))
                    # Block at the domain boundary
                    push!(conn, layer.level, Boundary(blk, s, bnd))
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

"""
    Fill ghost cells by copying neighboring cells in the same layer.
    `v` contains a vector with `Neighbor` relations.
"""
function fill_ghost_copy!(u::ScalarBlockField{D}, v::Vector{Neighbor{D}}) where {D}
    @batch for edge in v
        overlap = overlapindices(u, edge.face)
        ghost = ghostindices(u, -edge.face)
        
        ufrom = u[edge.from]
        uto = u[edge.to]
        
        for I in CartesianIndices(overlap)
            uto[ghost[I]] = ufrom[overlap[I]]
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
function fill_ghost_bnd!(u::ScalarBlockField{D}, v::Vector{Boundary{D}}, bc) where {D}
    @batch for link in v
        ghost = ghostindices(u, link.face)
        valid = mirrorghost(u, ghost, link.bnd)
        
        ublk = u[link.block]
        CI = CartesianIndices(ghost)
        # This is wrong for corners not in the domain corners.
        s = getbc(bc, link.bnd)
        
        for I in CI
            # I1 = __mirrorindex(I, CI, link.bnd)
            ublk[ghost[I]] = s * ublk[valid[I]]
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
function fill_ghost_interp!(u::ScalarBlockField{D}, v::Vector{RefBoundary{D}}) where {D}
    @batch for edge in v
        src = u[edge.coarse]
        dest = u[edge.fine]
        interp!(dest, ghostindices(u, edge.face),
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


@inline function __mirrorindex(I::CartesianIndex{D}, CI::CartesianIndices{D},
                        bnd::CartesianIndex{D}) where {D}
    CartesianIndex(ntuple(d -> (bnd[d] == 0 ? I[d] :
                                (last(CI.indices[d]) - I[d] + 1)), Val(D)))
end
