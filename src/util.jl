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
                    
                    v = u[blk]
                    #v[addghost(u, I)] = f(r...)
                    
                    # Indexing with CartesianIndex into an MArray falls back
                    # to an AbstractArray method which is much less efficient.
                    u.u[blk][Tuple(addghost(u, I))...] = @ncall $D f r
                end
            end
        end
    end
end


function maptree!(f::F, u::ScalarBlockField{D}, tree, h=1.0) where {D, F}
    @blocks order=flat for (l, coord, blk) in tree
    # for layer in tree
    #      l = layer.level
    #     for (coord, blk) in layer.pairs
        centers = cell_centers(coord, l, sidelength(u), h)
        for I in localindices(u)
            r = ntuple(d -> centers[d][I[d]], Val(D))
            v = u[blk]
            u.u[blk][Tuple(addghost(u, I))...] = f(r...)
        end
    end
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
