#= Utility & debug code 
=#


""" 
Populate a tree with all blocks up to level level.
"""
function populate!(tree)
    i = 0
    for layer in tree
        for c in layer.domain
            i += 1
            layer[c] = i
        end
    end

    sync!(tree)
    return i
end


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
@generated function maptree!(f, u::ScalarBlockField{D}, tree, h=1.0) where {D}
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
