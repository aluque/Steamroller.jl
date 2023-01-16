"""
Abstract type for refinement criteria.  Sub-types should implement

```
compatible(::Refinement, blk, I, h)
```
"""
abstract type AbstractRefinement end


"""
Test whether grid size `h` is compatible with refinement in block with `blk` at the location `I`.
"""
function compatible(::AbstractRefinement, blk, I, h)
    error("Subtypes of AbstractRefinement should implement compatible methods")
end


"""
A trivial refinement criterium that imposes a fixed maximum grid size.
"""
struct FixedRef{T} <: AbstractRefinement
    hmax::T
end

compatible(ref::FixedRef, blk, I, h) = h < ref.hmax



"""
A refinement criterium based on electron density.  We must refine if, when the electron density
is larger than nemax the grid size is larger than hmax. 
"""
struct ElectronDensityRef{T, SBF <: ScalarBlockField} <: AbstractRefinement
    ne::SBF
    nemax::T
    hmax::T
end

compatible(ref::ElectronDensityRef, blk, I, h) = (h < ref.hmax) || (ref.ne[I, blk] < ref.nemax)



"""
The refinement creiterium proposed in Teunnisen (2017), which depends on the electric field, `eabs`,
a transport model `transport` and three parameters `c0`, `c1` and `neighs` which is the 
number of neighboring cells where the refinement "spills".
"""
struct TeunissenRef{T, SBF <: ScalarBlockField, S <: AbstractTransportModel} <: AbstractRefinement
    eabs::SBF
    transport::S
    c0::T
    c1::T
end

function compatible(ref::TeunissenRef, blk, I, h)
    (;eabs, transport, c0, c1) = ref

    maxh = c0 * c1 / townsend(transport, c1 * eabs[I, blk])
    return h < maxh
end


"""
Mark cells that at a given level do not satisfy the refinement criterium.

The result is stored in `m` as true/false.
"""
@bkernel function refmark!((tree, lvl, coord, blk), m::ScalarBlockField{D},
                           ref::AbstractRefinement, h) where {D}    
    h /= (1 << lvl)
    
    m[blk] .= false

    # Perhaps we should find a way to stop working once we have found a positive cell
    # the problem is how to match this with the spilling.
    for I in validindices(m)
        m[I, blk] = !compatible(ref, blk, I, h)
    end
end

@enum RefDelta DEREFINE=-1 KEEP=0 REFINE=1
Base.zero(::Type{Steamroll.RefDelta}) = KEEP

"""
Look for the required changes in the tree according to the refining mark `m`
The result is stored in the refmark output which for each block will contain:

 -1: Block can be safely derefined (only meaningful if it is not a leaf)
 0: Block must be kept as is.
 1: Block must be refined.

"""
function refine!(tree, m::ScalarBlockField{D},
                 delta::ScalarBlockField{D},
                 stencil) where {D}
    delta .= DEREFINE
    
    @blocks order=serial for (lvl, c, blk) in tree        
        delta1 = delta[blk][]
        
        if isleaf(tree, lvl, c)
            if any(m[blk])
                delta[blk][] = REFINE

                if delta1 != REFINE
                    ensure_proper!(tree, delta, stencil, lvl, c)
                end
                keep_parent!(tree, delta, lvl, c)
            else
                delta[blk][] = max(KEEP, delta[blk][])
            end            
        else
            if any(m[blk])
                delta[blk][] = max(KEEP, delta[blk][])
            end

            # watch out! the KEEP state of delta may have been set by some other block,
            # not by the above if.
            if  delta[blk][] >= KEEP
                ensure_proper!(tree, delta, stencil, lvl, c)
            end

            keep_parent!(tree, delta, lvl, c)
        end
    end
end


"""
Make sure that the parent of a block in `tree` at level `lvl` and coords `c`
does not lose its parent.
"""
function keep_parent!(tree, delta, lvl, c)
    lvl == 1 && return nothing
    
    pc = parentcoord(c)
    pblk = get(tree[lvl - 1], pc)
    delta[pblk][] = max(delta[pblk][], KEEP)

    nothing
end



"""
Make sure that all neighbors of block with coordinates `c` at level `lvl` exist.
This should be called when the final state of `c` is a non-leaf.
"""
    
function ensure_proper!(tree, delta, stencil, lvl, c)
    for s in stencil
        if !((c + s) in tree[lvl].domain)
            continue
        end
        
        other = get(tree[lvl], c + s)
        
        if other == 0
            @assert lvl > 1
            
            # neighbor does not exist
            pc = parentcoord(c + s)
            pblk = get(tree[lvl - 1], pc)

            @assert pblk != 0 "Proper nesting broken at level=$(lvl) block coordinates $(Tuple(c + s))"
            delta1, delta[pblk][] = delta[pblk][], REFINE

            # if delta is already REFINE, we have already gone though this
            if delta1 != REFINE
                ensure_proper!(tree, delta, stencil, lvl - 1, pc)
            end
        else
            delta[other][] = max(delta[other][], KEEP)
        end
    end
end
