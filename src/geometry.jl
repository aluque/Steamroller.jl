#= 
Generic code for cylindrical / planar geometry.
=#

"""
A geometry is a type that allows us for example to implement
cylindrical symmetry in a generic code without a performance overhead.

Instances of this type must be able to compute from a given grid coordinate and
a stencil shift a factor for the laplacian term.
"""
abstract type AbstractGeometry end

struct CartesianGeometry <: AbstractGeometry end
@inline factor(::CartesianGeometry, c, d, ::Type{T}=Float64) where T = one(T)
factorexpr(::CartesianGeometry, c, d) = 1


"""
Number of radial cells covered by the coefficient tables of `CylindricalGeometry{D}()`.
"""
const DEFAULT_CYLINDRICAL_CELLS = 1 << 18

"""
Cylindrical geometry with radial dimension `D` (stored as a type parameter for performance).

The coefficients of the 2nd-order radial stencil are tabulated by global radial cell index `i`
(starting at 1 on the axis) to avoid two divisions per cell and sweep.  The tables cover
`n` radial cells; use `n = rootsize[D] * M * 2^(maxlevel - 1)`.  They are stored with
type `T`, which should be the field type (Float64 tables would promote the stencils).
"""
struct CylindricalGeometry{D, T} <: AbstractGeometry
    "(2i - 2) / (2i - 1): coefficient of the neighbor towards the axis"
    cm::Vector{T}

    "2i / (2i - 1): coefficient of the neighbor away from the axis"
    cp::Vector{T}

    function CylindricalGeometry{D, T}(n::Integer=DEFAULT_CYLINDRICAL_CELLS) where {D, T}
        new{D, T}([T((2i - 2) / (2i - 1)) for i in 1:n], [T(2i / (2i - 1)) for i in 1:n])
    end
end

CylindricalGeometry{D}(n::Integer=DEFAULT_CYLINDRICAL_CELLS) where D = CylindricalGeometry{D, Float64}(n)


"""
Coefficients `((2i - 2) / (2i - 1), 2i / (2i - 1))` of global radial cell index `i`.
Callers that read them inside `@inbounds` must check the whole block first with `checkblock`.
"""
@propagate_inbounds function radialcoeffs(geom::CylindricalGeometry, i)
    @boundscheck checkbounds(geom.cm, i)
    return @inbounds (geom.cm[i], geom.cp[i])
end


"""
Throw a `BoundsError` unless the tables of `geom` cover all valid cells of the block with
first global cell `gbl0` and size `M`.  Kernels call this once per block before the `@inbounds`
cell loop that evaluates stencils.
"""
checkblock(::AbstractGeometry, gbl0, M) = nothing
checkblock(geom::CylindricalGeometry{D}, gbl0, M) where D = checkbounds(geom.cm, gbl0[D]:(gbl0[D] + M - 1))


"""
Compute `r[i + δ/2]/r[i]` around coordinates given by `c`. r[i] is a cell 
center; r[i + 1/2] is the cell edge.  The result has type `T`.
"""
@inline function factor(::CylindricalGeometry{D}, c, δ, ::Type{T}=Float64) where {D, T}
     abs(1 + T(δ[D]) / T(2 * c[D] - 1))
end

# @inline function factor(::CylindricalGeometry{D}, c, d::Int) where D
#      1 + d / (2 * c[D] - 1)
# end

function factorexpr(::CylindricalGeometry{D}, c, d) where D
    :(1.0 + $(d[D]) / (2 * $(c[D]) - 1))
end

"""
Cell volume factor along dimension d of a cell with faces from f1 to f1 .+ h.  Along the radial
dimension of a cylindrical geometry it is π((f1 + h)^2 - f1^2) = 2π h (f1 + h / 2).
"""
@inline volfactor(::CartesianGeometry, d, f1, h) = h
@inline volfactor(::CylindricalGeometry{D}, d, f1, h) where D = d == D ? oftype(h, 2π) * h * (f1 + h / 2) : h


"""
Volume of the cell with local index `I` (1-based, no ghosts) in block `b` at level `l`, given `m`
cells per block.
"""
@inline function cellvolume(geom::AbstractGeometry, b::CartesianIndex{D}, I, l, m, h) where D
    hl = unitcell(l, h)
    f = cell_lower(b, I, l, m, h)
    prod(ntuple(d -> volfactor(geom, d, f[d], hl), Val(D)))
end

@inline cellvolume(::CartesianGeometry, b::CartesianIndex{D}, I, l, m, h) where D = unitcell(l, h)^D
