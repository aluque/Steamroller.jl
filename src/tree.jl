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
Add a block to the tree at level l, coordinate c and block index blk
"""
addblock!(tree, l, c, blk) = setindex!(tree[l], blk, c)

""" 
Populate a tree with all blocks up to level `level`.
"""
function populate!(tree, level=length(tree))
    @assert level <= length(tree)
    i = 0

    for l in 1:level        
        for c in tree[l].domain
            i += 1
            tree[l][c] = i
        end
    end

    sync!(tree)
    return i
end




"""
Return a range for blocks of all levels contained in the tree
"""
iblocks(t::Tree) = 1:nblocks(t)


"""
Return a tuple (level, coord, index) corresponding to block number `n` in the whole
tree.  This is useful sometimes as a way of traversing the tree in parallel but has
a small overhead.
"""
function nth(t::Tree{D}, n)::Tuple{Int64, CartesianIndex{D}, Int64} where D
    lvl = 1
    while lvl <= length(t)
        if n <= length(t[lvl])
            return (lvl, t[lvl].pairs[n]...)
        end
        n -= length(t[lvl])
        lvl += 1
    end

    error("Index out of the tree")
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

        if length(iter.tree) < lvl || length(iter.tree[lvl].pairs) == 0
            return nothing
        end
        
        i = 1
    end
    
    return ((lvl, iter.tree[lvl].pairs[i]...), (lvl, i + 1))
end

Base.length(iter::TreeIterator) = nblocks(iter.tree)

"""
Check if the block with coordinates `c` at level `lvl` is a leaf (does not have any children)
"""
function isleaf(t, lvl, B)
    for C in subblocks(B)
        # note that usually the block is always refined completely so we could check only the
        # existence of the first child.  We check all of them to make this robust to other
        # refinement schemes.
        if hasblock(t[lvl + 1], C)
            return false
        end
    end
    return true
end


"""
Iterate over all blocks in a tree with a mapreduce operation. The map operation `f` receives
arguments `(level, coordinates, blk)`.  The reduction receives whatever `f` returns and initially
`init` (which defaults to `nothing`).

NOTE: We do not add a method to Base.mapreduce because `Tree` is defined as a `Vector{BlockLayer}`. 
"""
function mapreduce_tree(f, op, tree::Tree, init=nothing)
    x = init
    for (lvl, I, blk) in TreeIterator(tree)
        x = op(x, f(lvl, I, blk))
    end

    return x
end
