#= 
These are lowest level mesh-handling methods.  Some points to keep in mind:

1. We think in terms of two meshes: the 'block' mesh and the 'cell' mesh.
   Here we assume that these meshes are infinite.
3. Each block contains m cells.
4. "global" cell coordinates are those that index the full mesh.
   "local" coordinates are within a block.
4. We use b to index blocks, i for cells.  These are CartesianIndex{D}, where
   D is the number of dimensions (usually 2 or 3).
5. Many functions admit an h parameter that is the size of a cell at level 1.
   It defaults to 1.0.
=#


@inline unitcell(l, h=1.0) = h / (1 << (l - 1))

"""
Bounding box of a block at level `l` with `m` cells per block.
"""
@inline function bbox(b::CartesianIndex{D}, l, m, h=1.0) where {D}
    L = m * unitcell(l, h)

    x1 = ntuple(d -> (L * (b[d] - 1), L * b[d]), Val(D))
end

"""
Children of blocks.
"""
@inline function subblocks(b::CartesianIndex{D}) where {D}
    ranges = CartesianIndices(ntuple(d -> (2 * (b[d] - 1) + 1):(2 * b[d]), Val(D)))
end


""" 
Return coordinate of the parent block. 
"""
function parentcoord(c::CartesianIndex{D}) where {D}
    CartesianIndex(ntuple(d -> div(c[d] - 1, 2) + 1, Val(D)))
end


""" 
Return sub-coordinate of this block within the parent. 

This is the coordinates consisting in 1s and 2s that specifies which subblock
in the parent corresponds to this block.
"""
function subcoord(c::CartesianIndex{D}) where {D}
    CartesianIndex(ntuple(d -> rem(c[d] - 1, 2) + 1, Val(D)))
end

"""
Blocks that are children of the same parent as `c`
"""
@inline function siblings(c::CartesianIndex{D}) where {D}
    CartesianIndices(ntuple(d -> (2 * fld(c[d] - 1, 2) + 1):(2 * fld(c[d] - 1, 2) + 2), Val(D)))
end



"""
Return coordinates of the first cell (e.g. lower left) in a block as
global coordinates.
"""
@inline function global_first(b, m)
    m * (b - oneunit(b)) + oneunit(b)
end


"""
Return coordinates of the last cell (e.g. upper right) in a block.
"""
@inline function global_last(b, m)
    m * b
end


"""
Return cells contained in block `b` given `m` cells per block given
as CartesianIndices
"""
@inline function global_indices(b, m)
    i0 = global_first(b, m)
    i1 = global_last(b, m)
    i0:i1
end


"""
Cell centers in block `b` at level `l` given `m` cells per block.
"""
@inline function cell_centers(b::CartesianIndex{D}, l, m, h=1.0) where {D}
    inds = global_indices(b, m)
    hl = unitcell(l, h)
    
    ntuple(d -> hl * inds.indices[d] .- (hl / 2), Val(D))
end

