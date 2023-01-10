#=
Macros to make it easier to walk the tree

The main issue here is that it is hard to handle macro hygiene with macros calling macros so we have
to do some hacking to allow @batch inside the macros.
=#


"""
A macro equivalent to @batch per=thread from Polyester.jl.
"""
macro tbatch(ex)
    reserve, minbatch, per, threadlocal = 0, 1, :thread, (Symbol(""), :Any)
    Polyester.enclose(macroexpand(__module__, ex), reserve, minbatch, per, threadlocal, __module__)
end

"""
Iterate in parallel over all blocks at all levels in a tree.

This macro must prefix a for loop as, for example
```julia
@blocks for (level, coord, blk) in tree
    println("block index \$(blk) is at level \$(level) and coordinates \$(coord)")
end
```

The loop variables must always be tree in a tuple, for level, coordinate and block index.

Optionally you can specify order=deepfirst or order=flat after the @blocks macro.  For example:

```julia
@blocks order=deepfirst for (level, coord, blk) in tree
    println("block index \$(blk) is at level \$(level) and coordinates \$(coord)")
end
```

`deepfirst` indicates that the iteration will proceed from the higher (finest) level and will not
move to a lower level until the each level is exhausted.  Use this if the operation in the loop at
one level depends on higher levels.  Within each level the order of blocks is non-deterministic.

`flat` indicates that the operation can performed on blocks at an arbitrary order. This is generally
more efficient as blocks are divided into bunches and allocated to threads more symetrically.
"""
macro blocks(ex1, ex2)
    ok = @capture(ex1, order=order_)
    if !ok
        error("Two-argument @blocks must receive order=deepfirst or order=flat as first arguments")
    end

    ok = @capture(ex2, for (vars__,) in tree_ body_ end)

    if !ok
        error("Incorrect syntax for the @blocks macro")
    end

    if order == :deepfirst
        out = deepfirst(vars, tree, body)
    elseif order == :flat
        out = flat(vars, tree, body)
    else
        error("Order $(order) unknown")
    end

    esc(out)
end

function deepfirst(vars, tree, body)
    symlevel, symcoord, symindex = vars

    inner = :(for ($symcoord, $symindex) in $tree[$symlevel].pairs
            $body
        end)
    
    binner = macroexpand(@__MODULE__, :(Polyester.@batch(per=thread, $inner)))
    
    outer = quote
        # Ensure inverse tree enumeration
        for $symlevel in length($tree):-1:1
            $inner
        end
    end

    return outer
end

function flat(vars, tree, body)
    symlevel, symcoord, symindex = vars
    i = gensym(:blocks)

    loop = :(for $i in iblocks($tree)
             ($symlevel, $symcoord, $symindex) = nth($tree, $i)
             $body
             end)
    @show loop
    bloop = macroexpand(@__MODULE__, :(Polyester.@batch(per=thread, $loop)))
    
    return bloop
end


# function blockbased(ex)
#     ex1 = MacroTools.longdef(ex)
#     dict = splitdef(ex1)
#     args = map(splitarg, dict[:args])
#     nbarg = findfirst(arg -> arg[2] == :BlockData, args)
#     !isnothing(nbarg) || error("One of the arguments must have type ::BlockData")

#     argswithlevel = map(
#     quote
#         ex
#     end
        
# end

    
