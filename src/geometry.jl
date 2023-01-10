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
@inline factor(::CartesianGeometry, c, d) = 1
factorexpr(::CartesianGeometry, c, d) = 1


# For performance we store the cylindrical dimension in a type parameters
struct CylindricalGeometry{D} <: AbstractGeometry end


"""
Compute `r[i + δ/2]/r[i]` around coordinates given by `c`. r[i] is a cell 
center; r[i + 1/2] is the cell edge.
"""
@inline function factor(::CylindricalGeometry{D}, c, δ) where D
     abs(1 + δ[D] / (2 * c[D] - 1))
end

# @inline function factor(::CylindricalGeometry{D}, c, d::Int) where D
#      1 + d / (2 * c[D] - 1)
# end

function factorexpr(::CylindricalGeometry{D}, c, d) where D
    :(1.0 + $(d[D]) / (2 * $(c[D]) - 1))
end

