#=
The mesh is organized in layers for each level, each of them represented by a 
BlockLayer object.
=#

"""
    Block layer in a space of dimension `D`.
"""
struct BlockLayer{D}
    level::Int

    # The full domain (in blocks, at this level)
    domain::CartesianIndices{D, NTuple{D, UnitRange{Int}}}
    
    # Map of coordinates -> block index
    index::Dict{CartesianIndex{D}, BlockIndex}

    # Map of block index -> coordinates
    coord::Dict{BlockIndex, CartesianIndex{D}}

    # For fast iteration we store a vector with pairs index => coord.    
    # this must be updated with sync! whenever the tree changes.
    pairs::Vector{Pair{CartesianIndex{D}, BlockIndex}}
end


function Base.show(io::IO, layer::BlockLayer)
    print(io, "layer at level = ", layer.level, ", with blocks:\n")
    for (coord, blk) in pairs(layer.index)
        print(io, "   ", coord, " => ", Int(blk), "\n")
    end
end


function BlockLayer{D}(level, domain) where {D}
    index = Dict{CartesianIndex{D}, BlockIndex}()
    coord = Dict{BlockIndex, CartesianIndex{D}}()
    pairs = Pair{CartesianIndex{D}, BlockIndex}[]
    
    BlockLayer(level, domain, index, coord, pairs)
end


""" 
    Synchronize the pairs list with the index and coord dicts.
    Must be called after each update of the layer (tree).
"""
function sync!(layer::BlockLayer)
    empty!(layer.pairs)
    for (coord, blk) in layer.index
        push!(layer.pairs, coord => blk)
    end
end


Base.getindex(layer::BlockLayer, coord::CartesianIndex) = getindex(layer.index, coord)
Base.get(layer::BlockLayer, coord::CartesianIndex, default::BlockIndex=zero(BlockIndex)) = get(layer.index, coord, default)
# Base.getindex(layer::BlockLayer, blk::BlockIndex) = getindex(layer.coord, blk)


# Iteriation is passed down to a dictionary iterator
Base.iterate(layer::BlockLayer) = iterate(layer.pairs)
Base.iterate(layer::BlockLayer, state) = iterate(layer.pairs, state)
Base.length(layer::BlockLayer) = length(layer.pairs)
Base.eltype(layer::BlockLayer) = eltype(layer.pairs)


""" 
    Checks if a block exists in `layer` at a given coordinate `coord`.
"""
hasblock(layer, coord) = (coord in keys(layer.index))


"""
    Add a block to a block layer `layer` at location `coord` with index `blk`.
"""
function Base.setindex!(layer::BlockLayer, blk, coord)
    # Make sure that no duplicated id is added
    @assert !(coord in keys(layer.index)) "Cannot overwrite existing block"
    @assert !(blk in keys(layer.coord)) "Cannot overwrite existing block"
    
    layer.index[coord] = blk
    layer.coord[blk] = coord
end


"""
    Remove from `layer` the block at location `coord`.
"""
function Base.delete!(layer::BlockLayer, coord::CartesianIndex)
    blk = get(layer.index, coord, 0)
    @assert blk != 0 "Trying to remove nonexisting block"
    
    delete!(layer.index, coord)
end


"""
    Remove from `layer` the block with index idx.
"""
function Base.delete!(layer::BlockLayer{D}, blk::BlockIndex) where D
    coord = get(layer.coord, blk, nothing)
    @assert !isnothing(coord) "Trying to remove nonexisting block"
    
    delete!(layer.coord, blk)
end


multiplyrange(r, factor) = ((first(r) - 1) * factor + 1) : (factor * last(r))

function multiplyindices(inds::CartesianIndices{D}, factor) where {D}
    CartesianIndices(ntuple(d -> multiplyrange(inds.indices[d], factor), Val(D)))
end


""" 
    `Tree{D}` is a tree embedded in a space with dimension `D`.

    The tree structure is represented by a `Vector` of `BlockLayers`.
"""
const Tree{D} = Vector{BlockLayer{D}}


"""
    Create a series of layers up to level lmax, starting at level 1 with
    a cube with sides of m blocks in dimension d.
"""
function Tree(D, domain, levels)
    map(l -> BlockLayer{D}(l, multiplyindices(domain, 1 << (l - 1))), 1:levels)
end

"""
    Return the number of blocks in the tree.
"""
nblocks(t::Tree) = sum(layer -> length(layer), t)


"""
    Return a range for blocks of all levels contained in the tree
"""
iblocks(t::Tree) = 1:nblocks(t)


"""
    Return a tuple (level, coord, index) corresponding to block number `n` in the whole
    tree.  This is useful sometimes as a way of traversing the tree in parallel but has
    a small overhead.
"""
function nth(t::Tree, n)
    lvl = 1
    while lvl <= length(t)
        if n <= length(t[lvl])
            return (lvl, t[lvl].pairs[n]...)
        end
        n -= length(t[lvl])
        lvl += 1
    end
end


dimension(::Tree{D}) where {D} = D

sync!(tree::Tree) = foreach(sync!, tree)

# TreeIterator : returns tuples (level, coord, index).
# The problem with using this iterator is that it's not parallelizable.
struct TreeIterator{D}
    tree::Tree{D}
end

function Base.iterate(iter::TreeIterator, state=(1, 1))
    (lvl, i) = state

    isempty(iter.tree) && (return nothing)
    
    if length(iter.tree[lvl].pairs) < i
        lvl += 1

        if length(iter.tree) < lvl
            return nothing
        end
        
        i = 1
    end
    
    return ((lvl, iter.tree[lvl].pairs[i]...), (lvl, i + 1))
end

Base.length(iter::TreeIterator) = nblocks(iter.tree)
