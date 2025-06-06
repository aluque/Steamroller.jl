"""
Abstract type for refinement criteria.  Sub-types should implement

```
compatible(::Refinement, blk, I, h)
```
"""
abstract type AbstractRefinement end

"""
Set a decayed refinement.  By default, refinement decays instantaneously.
"""
decay(ref::AbstractRefinement, v, dt) = zero(v)

"""
Test whether grid size `h` is compatible with refinement in block with `blk` at the location `I`.
"""
function compatible(::AbstractRefinement, (lvl, coord, blk), I, h, t)
    error("Subtypes of AbstractRefinement should implement compatible methods")
end


"""
A trivial refinement criterium that imposes a fixed maximum grid size.
"""
struct FixedRef{T} <: AbstractRefinement
    hmax::T
end

compatible(ref::FixedRef, (lvl, coord, blk), I, h, t) = h < ref.hmax


"""
A refinement criterium based on electron density.  We must refine if, when the electron density
is larger than nemax the grid size is larger than hmax. 
"""
struct ElectronDensityRef{T, SBF <: ScalarBlockField} <: AbstractRefinement
    ne::SBF
    nemax::T
    hmax::T
end

compatible(ref::ElectronDensityRef, (lvl, coord, blk), I, h, t) = (h < ref.hmax) || (ref.ne[addghost(ref.ne, I), blk] < ref.nemax)


"""
A refinement criterium based on one dimension (generally radius).  We must refine if, when the
dimension is smaller than threshold, the grid size is larger than hmax. 
"""
struct DirThresholdRef{Dir, M, T} <: AbstractRefinement
    threshold::T
    hmax::T
end

function compatible(ref::DirThresholdRef{Dir, M, T}, (lvl, coord, blk), I, h, t) where {Dir, M, T}
    (h < ref.hmax) && (return true)
    gbl0 = global_first(coord, M)
    
    return h * (I[Dir] + gbl0[Dir] - 1 - convert(T, 1/2)) > ref.threshold    
end


"""
A refinement criterium based on density.  We must refine if, when the density
is larger than nemax the grid size is larger than hmax. 
"""
struct DensityRef{T, SBF <: ScalarBlockField} <: AbstractRefinement
    nh::SBF
    nhmax::T
    hmax::T
end

compatible(ref::DensityRef, (lvl, coord, blk), I, h, t) = (h < ref.hmax) || (ref.nh[addghost(ref.nh, I), blk] < ref.nhmax)


"""
The refinement criterium proposed in Teunnisen (2017), which depends on the electric field, `eabs`,
a transport model `transport` and two parameters `c0`, `c1`.
"""
struct TeunissenRef{T, SBF <: ScalarBlockField,
                    S <: AbstractTransportModel, DS <: AbstractDensityScaling} <: AbstractRefinement
    eabs::SBF
    transport::S
    c0::T
    c1::T
    dens::DS
end

TeunissenRef(eabs, transport, c0, c1) = TeunissenRef(eabs, transport, c0, c1, TrivialDensityScaling())
    
function compatible(ref::TeunissenRef, (lvl, coord, blk), I, h, t)
    (;eabs, transport, dens, c0, c1) = ref
    I1 = addghost(eabs, I)
    gbl0 = global_first(coord, sidelength(ref.eabs))
    x = ntuple(d -> h * (gbl0[d] + I[d] - 1) - h / 2, Val(dimension(ref.eabs)))
    theta = nscale(dens, x)
    maxh = c0 * c1 / (townsend(transport, c1 * eabs[I1, blk] / theta) * theta)
    return h < maxh
end


"""
A refinement creiterium based on the value of the laplacian applied to a given field.
Strictly, we compare Lu/u with a fixed threshold alpha, where L is the discrete laplacian operator with
h=1.
"""
struct LaplacianRef{T, SBF <: ScalarBlockField, GEOM <: AbstractGeometry, L <: LaplacianDiscretization}
    alpha::T
    field::SBF
    lpl::L    
    geom::GEOM
end

function compatible(ref::LaplacianRef, (lvl, coord, blk), I, h, t)
    (;alpha, field, lpl, geom) = ref
    M = sidelength(field)
    J = global_first(coord, M)
    I1 = addghost(field, I)
    c = applystencil(field[blk], 0.0, I1, J, lpl, geom, Val{:lhs}()) / field[I1, blk]
    # if abs(c) > 0.1 && lvl > 4
    #     @show c alpha lvl coord I
    # end
    
    return abs(c) < alpha
end


"""
A refinement criterium that combines two criteria in an "and" relationship, i.e. a cell
is `compatible` if it is compatible according to criterium 1 and criterium 2.
"""
struct AndRef{R1, R2} <: AbstractRefinement
    crit1::R1
    crit2::R2
end

function compatible(ref::AndRef, (lvl, coord, blk), I, h, t)
    return compatible(ref.crit1, (lvl, coord, blk), I, h, t) && compatible(ref.crit2, (lvl, coord, blk), I, h, t)
end


"""
A refinement criterium that combines two criteria in an "or" relationship, i.e. a cell
is `compatible` if it is compatible according to criterium 1 or criterium 2.
"""
struct OrRef{R1, R2} <: AbstractRefinement
    crit1::R1
    crit2::R2
end

function compatible(ref::OrRef, (lvl, coord, blk), I, h, t)
    return compatible(ref.crit1, (lvl, coord, blk), I, h, t) || compatible(ref.crit2, (lvl, coord, blk), I, h, t)
end


"""
A refinement criterium that is active only in a given time interval [`tmin`, `tmax`).  Outside this interval
every cell is `compatible`.
"""
struct TimeLimitedRef{T, R} <: AbstractRefinement
    tmin::T
    tmax::T
    crit::R
end

function compatible(ref::TimeLimitedRef, (lvl, coord, blk), I, h, t)
    (;tmin, tmax, crit) = ref
    (tmin <= t < tmax) || return true
    
    return compatible(crit, (lvl, coord, blk), I, h, t)
end

"""
A refinement with some persistence in time.
"""
struct PersistingRef{T, R} <: AbstractRefinement
    decay::T
    crit::R
end

compatible(ref::PersistingRef, args...) = compatible(ref.crit, args...)
decay(ref::PersistingRef, v, dt) = v > 0 ? v - dt / ref.decay : zero(v)


"""
Mark cells that at a given level do not satisfy the refinement criterium.

The result is stored in `m` as true/false.
"""
@bkernel function refmark!((tree, lvl, coord, blk), m::ScalarBlockField{D},
                           ref::AbstractRefinement, h, t, dt) where {D}    
    h /= (1 << (lvl - 1))
    
    # m[blk] .= false

    # Perhaps we should find a way to stop working once we have found a positive cell
    # the problem is how to match this with the spilling.
    for I in localindices(m)
        if !compatible(ref, (lvl, coord, blk), I, h, t)
            m[addghost(m, I), blk] = 1
        else
            m[addghost(m, I), blk] = decay(ref, m[addghost(m, I), blk], dt)
        end
    end
end

@enum RefDelta REMOVE=-1 KEEP=0 REFINE=1
Base.zero(::Type{Steamroller.RefDelta}) = KEEP
Base.convert(::Type{Float64}, d::RefDelta) = convert(Float64, Int(d))

"""
Look for the required changes in the tree according to the refining mark `m`
The result is stored in the `delta` output which for each block will contain:

 -1: Block can be safely derefined (only meaningful if it is not a leaf)
 0: Block must be kept as is.
 1: Block must be refined.

"""
function refdelta!(tree, delta::ScalarBlockField{D},
                   m::ScalarBlockField{D},
                   stencil) where {D}
    delta .= KEEP
    # First pass: we look only at each block independently of its neighbors
    @blocks order=flat for (lvl, c, blk) in tree        
        if any(>(0), m[blk])
            delta[blk][] = REFINE
        else
            if isleaf(tree, lvl, c)
                delta[blk][] = REMOVE
            end
        end
    end

    # ensure proper nesting
    @blocks order=serial for (lvl, c, blk) in tree        
        isleaf(tree, lvl, c) || continue        
        if delta[blk][] == REFINE
            ensure_proper_refine!(tree, delta, stencil, lvl, c)
        else
            ensure_proper!(tree, delta, stencil, lvl, c)
        end        
    end

    # ensure that only remove blocks if all siblings are marked for removal
    @blocks order=serial for (lvl, c, blk) in tree        
        isleaf(tree, lvl, c) || continue

        if delta[blk][] == REMOVE            
            pc = parentcoord(c)
            pblk = get(tree[lvl - 1], pc)

            if delta[pblk][] == REFINE
                delta[blk][] = KEEP
            else
                for sib in siblings(c)
                    sblk = tree[lvl][sib]
                    delta[blk][] = max(delta[blk][], delta[sblk][])
                end
            end
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
"""
function ensure_proper_refine!(tree, delta, stencil, lvl, c)
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
            delta[pblk][] = REFINE

            # if delta is already REFINE, we have already gone though this
            # if delta1 != REFINE
            #     ensure_proper!(tree, delta, stencil, lvl - 1, pc)
            # end
        else
            delta[other][] = max(delta[other][], KEEP)
        end
    end
end

"""
Make sure that all (possible non-filled) neighbors of a block have parents.  This should be called
for all leaf nodes to ensure proper refinement.
"""
function ensure_proper!(tree, delta, stencil, lvl, c)
    for s in stencil
        if !((c + s) in tree[lvl].domain)
            continue
        end
        
        pc = parentcoord(c + s)
        pblk = get(tree[lvl - 1], pc)

        @assert pblk != 0 "Proper nesting broken at level=$(lvl) block coordinates $(Tuple(c + s))"

        delta[pblk][] = max(delta[pblk][], KEEP)
    end
end


"""
Apply to `tree` the refinement delta indicated by `refdelta`.

`freeblocks` is a list of availabe free blocks (which may be modified by this function).
"""
function applydelta!(tree, refdelta, freeblocks, fields, minlevel, maxlevel)
    @blocks order=serial for (lvl, coord, blk) in tree
        lvl > 1 || continue
        isleaf(tree, lvl, coord) || continue        
        
        if refdelta[blk][] == REMOVE && lvl > minlevel
            pc = parentcoord(coord)
            pblk = get(tree[lvl - 1], pc)

            restrict!(fields, pblk, blk, subcoord(coord))
            delete!(tree[lvl], coord)
            delete!(tree[lvl], blk)
            push!(freeblocks, blk)

        elseif refdelta[blk][] == REFINE && lvl < maxlevel
            for c in subblocks(coord)
                if isempty(freeblocks)
                    nblk = newblock!(fields)
                else
                    nblk = pop!(freeblocks)
                end
                
                # new blocks will not be included in the loop until the are synced
                addblock!(tree, lvl + 1, c, nblk)
                # pseudo-code
                interp!(fields, nblk, blk, subcoord(c))                
            end
        end
    end

    sync!(tree)
end
