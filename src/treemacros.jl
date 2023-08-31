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

Optionally you can specify order=[deepfirst|flat|serial] after the @blocks macro.  For example:

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

`serial` indicates that the operation should not run in parallel. It also guarantees that deeper
blocks are processed earlier.

"""
macro blocks(ex1, ex2=nothing)
    if isnothing(ex2)
        ex2 = ex1
        ex1 = :(order=flat)
    end

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
    elseif order == :serial
        out = deepfirst(vars, tree, body, false)
    else
        error("Order $(order) unknown")
    end

    esc(out)
end

# macro blocks(ex)
#     macroexpand(__module__, :(@blocks(order=flat, $ex)))
# end


function deepfirst(vars, tree, body, parallel=true)
    symlevel, symcoord, symindex = vars

    inner = :(for ($symcoord, $symindex) in $tree[$symlevel].pairs
            $body
        end)
    
    if parallel
        inner = macroexpand(@__MODULE__, :(Polyester.@batch(per=core, $inner)))
    end
    
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
    bloop = macroexpand(@__MODULE__, :(Polyester.@batch(per=core, $loop)))
    
    return bloop
end


"""
Define a function that can be used as a kernel that acts on all blocks of a tree.

This macro must operate on the definition of a function that takes as first argument
a tuple of (tree, level, coord, index) for individual blocks.  The macro then creates an additional
method of the function that can be invoked with a `::Tree` as first argument and will loop in 
parallel over all blocks in the tree.

```julia
@bkernel function kern((tree, level, coord, index), args...)
    ...
end

...

kern(tree, args...)

```

Optionally this macro accepts an `order=` parameter.  See the documentation for `@blocks` for
more detais.
"""
macro bkernel(ex, ex2=nothing)
    if !isnothing(ex2)
        ex, ex2 = ex2, ex
    else
        ex2 = :(order=flat)
    end
    
    isdef(ex) || error("@bkernel must act on a function definition")
    d = splitdef(ex)
    args = d[:args]

    if !(@capture(args[1], (t_, l_, c_, i_)))
        error("To use @bkernel the first argument must be a tuple (tree, level, coord, index)")
    end


    # We must strip the ::Type qualificatios because otherwise the variables may be passed
    # as abstract types and spoil performance.
    cargs = map(a -> a isa Symbol ? a : a.args[1], d[:args][2:end])
    ckwargs = map(a -> a isa Symbol ? a : a.args[1], d[:kwargs][2:end])
    
    loop = :(for ($l, $c, $i) in $t
                $(d[:name])(($t, $l, $c, $i), $(cargs...); $(ckwargs...))
             end
             )

    q = macroexpand(__module__, :(@blocks($ex2, $loop)))
    rx = :(function $(d[:name])($(t)::Tree, $(d[:args][2:end]...);
                          $(d[:kwargs]...)) where {$(d[:whereparams]...)}
           $q
           end)

    blk = quote
        Base.@__doc__ $(combinedef(d))
        $(rx)
    end |> esc
    return blk
end




# end

    
