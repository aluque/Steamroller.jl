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
function Base.findmax(u::ScalarBlockField{D}, tree, h=1.0) where D
    function blkmax(l, c, blk)
        centers = cell_centers(c, l, sidelength(u), h)
        I = argmax(valid(u, blk))
        return (u[I, blk], ntuple(d -> centers[d][I[d]], Val(D)))
    end

    function op((e1, r1), (e2, r2))
        return e1 > e2 ? (e1, r1) : (e2, r2)
    end
    
    mapreduce_tree(blkmax, op, tree, (typemin(eltype(u)), ntuple(d -> convert(eltype(u), NaN), Val(D))))
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
