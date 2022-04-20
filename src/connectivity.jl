#= 
Handling of connectivity: the spatial relationships between blocks and between blocks and boundaries.
=#

# The block is at a domain boundary
struct Boundary{D}
    block::BlockIndex
    face::CartesianIndex{D}
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
    Connectivity(fill(Boundary{D}[], levels),
                 fill(Neighbor{D}[], levels),
                 fill(RefBoundary{D}[], levels),
                 fill(Child{D}[], levels))
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
            for d in stencil
                if !((c + d) in layer.domain)
                    # Block at the domain boundary
                    push!(conn, layer.level, Boundary(blk, d))
                    continue
                end

                other = get(layer, c + d)
                if other != 0
                    # A neighbor exists
                    push!(conn, layer.level, Neighbor(blk, other, d))
                elseif layer.level > 1
                    # A refinement boundary
                    pc = parentcoord(c + d)
                    pblk = get(tree[layer.level - 1], pc)

                    @assert pblk != 0 "Proper nesting broken" 

                    push!(conn, layer.level,
                          RefBoundary(blk, pblk, d, subcoord(c + d)))
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


"""
    Fill ghost cells by copying neighboring cells in the same layer.
    `v` contains a vector with `Neighbor` relations.
"""
function fill_ghosts_copy!(u::ScalarBlockField{D}, v::Vector{Neighbor{D}}) where {D}
    @batch for edge in v
        overlap = overlapindices(u, edge.face)
        ghost = ghostindices(u, -edge.face)
        
        ufrom = getblk(u, edge.from)
        uto = getblk(u, edge.to)
        
        for I in CartesianIndices(overlap)
            uto[ghost[I]] = ufrom[overlap[I]]
        end
    end
end


"""
    Copy ghost cells from neighbouring blocks according to a full connectivity
    structure `conn`.
"""
function fill_ghosts_copy!(u::ScalarBlockField, conn::Connectivity)
    # For each layer...
    for v in conn.neighbor
        fill_ghosts_copy!(u, v)
    end
end


"""
    Fill ghost cells in the boundary of a given layer by applying the 
    boundary conditions specified by `bc`
"""
function fill_ghosts_bnd!(u::ScalarBlockField{D}, v::Vector{Boundary{D}}, bc) where {D}
    @batch for bnd in v
        overlap = overlapindices(u, bnd.face)
        ghost = ghostindices(u, bnd.face)

        ublk = getblk(u, bnd.block)
        CI = CartesianIndices(overlap)
        # This is wrong for corners not in the domain corners.
        s = getbc(bc, bnd.face)
        
        for I in CI
            I1 = __mirrorindex(I, CI, bnd.face)
            ublk[ghost[I]] = s * ublk[overlap[I1]]
        end
    end
end


"""
    Fill ghost cells in boundary blocks according to given boundary conditions 
    `bc`.
    structure `conn`.
"""
function fill_ghosts_bnd!(u::ScalarBlockField, conn::Connectivity, bc)
    # For each layer...
    for v in conn.boundary
        fill_ghosts_bnd!(u, v, bc)
    end
end


@inline function __mirrorindex(I::CartesianIndex{D}, CI::CartesianIndices{D},
                        face::CartesianIndex{D}) where {D}
    CartesianIndex(ntuple(d -> (face[d] == 0 ? I[d] :
                                (last(CI.indices[d]) - I[d] + 1)), Val(D)))
end
