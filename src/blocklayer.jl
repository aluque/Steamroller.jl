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
function Base.setindex!(layer::BlockLayer, blk, coord, overwrite=true)
    if !overwrite
        # Make sure that no duplicated id is added
        @assert !(coord in keys(layer.index)) "Cannot overwrite existing block"
        @assert !(blk in keys(layer.coord)) "Cannot overwrite existing block"
    end
    
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


