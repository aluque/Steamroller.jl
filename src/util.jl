#= Utility & debug code 
=#


"""
Apply the function `f(level, coord, blk)` to each block inside `tree`.

Note: Blocks may run in parallel.
"""
function foreachblock(f, tree)
    for layer in tree        
        # For parallel run we must pre-allocate the blocks not sure about the
        # performance impact of this
        @batch for (coord, blk) in layer.pairs
            f(layer.level, coord, blk)
        end
    end
end


function foreachblock_serial(f, tree)
    for layer in tree
        for (coord, blk) in layer.pairs
            f(layer.level, coord, blk)
        end
    end
end


"""
Map the function `f`, applied at cell centers, into all blocks of a 
`ScalarBlockField` `u`.
"""
@generated function maptree1!(f, u::ScalarBlockField{D}, tree, h=1.0) where {D}
    quote
        for layer in tree
            for (coord, blk) in layer.index
                l = layer.level
                centers = cell_centers(coord, l, sidelength(u), h)
                for I in localindices(u)
                    @nexprs $D d -> (r_d = centers[d][I[d]])
                    
                    u[blk][Tuple(addghost(u, I))...] = @ncall $D f r
                end
            end
        end
    end
end


function maptree!(f::F, u::ScalarBlockField{D}, tree, h=1.0) where {D, F}
    @blocks order=flat for (l, coord, blk) in tree
        centers = cell_centers(coord, l, sidelength(u), h)
        for I in localindices(u)
            r = ntuple(d -> centers[d][I[d]], Val(D))
            u[blk][Tuple(addghost(u, I))...] = f(r...)
        end
    end
end

"""
Find the maximum value of a field and return a tuple `(location, value)`.
"""
function findmaxloc(u::ScalarBlockField{D}, tree, h=1.0) where D
    function blkmax(l, c, blk)
        centers = cell_centers(c, l, sidelength(u), h)
        v = valid(u, blk)
        I = argmax(v)
        return (v[I], ntuple(d -> centers[d][I[d]], Val(D)))
    end

    function op((e1, r1), (e2, r2))
        return e1 > e2 ? (e1, r1) : (e2, r2)
    end
    
    mapreduce_tree(blkmax, op, tree, (typemin(eltype(u)), ntuple(d -> convert(eltype(u), NaN), Val(D))))
end


"""
Find the minimum value of a field and return a tuple `(location, value)`.
"""
function findminloc(u::ScalarBlockField{D}, tree, h=1.0) where D
    function blkmin(l, c, blk)
        centers = cell_centers(c, l, sidelength(u), h)
        v = valid(u, blk)
        I = argmin(v)
        return (v[I], ntuple(d -> centers[d][I[d]], Val(D)))
    end

    function op((e1, r1), (e2, r2))
        #return e1 < e2 ? (e1, r1) : (e2, r2)
        if e1 < e2
            return (e1, r1)
        else
            return (e2, r2)
        end        
    end
    
    mapreduce_tree(blkmin, op, tree, (typemax(eltype(u)), ntuple(d -> convert(eltype(u), NaN), Val(D))))
end

"""
Find the block and local coordinates (without ghosts) of the smallest cell containing a point `r`.
Returns a tuple `(blk, m, l)` where `blk` is the block index, `m` is the local coordinate
(CartesianIndex, 1-based, no ghost offset) of the containing cell inside the block and l is the
level. Returns `nothing` if the point is outside the tree.
"""
function treelocate(tree::Tree{D}, r, M, h=1.0) where D
    flev = finestlevel(tree)
    # Using 1 + ... and not fld1, mod1 because e.g. mod1(0.0, 8.0) == 8.0.
    c = CartesianIndex(ntuple(d -> round(Int, 1 + fld(r[d], unitcell(flev, h))), Val(D)))
    blk = zero(BlockIndex)
    for l in flev:-1:1
        coord = CartesianIndex(ntuple(d -> fld1(c[d], M), Val(D)))
        blk = get(tree[l], coord)
        if !iszero(blk)
            m = CartesianIndex(ntuple(d -> mod1(c[d], M), Val(D)))
            return (blk, m, l)
        end
        c = CartesianIndex(ntuple(d -> fld1(c[d], 2), Val(D)))
    end

    return nothing
end


# Formatting function to include times in the logs.
function metafmt(level::LogLevel, _module, group, id, file, line)
    @nospecialize
    color = Logging.default_logcolor(level)
    dt = format("[{:<23}] ", string(Dates.now()))
    
    prefix = string(dt, level == Logging.Warn ? "Warning" : string(level), ':')
    suffix::String = ""
    Logging.Info <= level < Logging.Warn && return color, prefix, suffix

    if level < Logging.Info
        return :blue, "", ""
    end
    
    _module !== nothing && (suffix *= "$(_module)")
    if file !== nothing
        _module !== nothing && (suffix *= " ")
        suffix *= Base.contractuser(file)::String
        if line !== nothing
            suffix *= ":$(isa(line, UnitRange) ? "$(first(line))-$(last(line))" : line)"
        end
    end
    !isempty(suffix) && (suffix = "@ " * suffix)
    return color, prefix, suffix
end
