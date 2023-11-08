#=
Homogeneous boundary conditions are represented by a 3x3 or 3x3x3 StaticArray.  

Each element stores +1/-1 for Neumann / Dirchlet b.c.; the center element
is ignored.

The support for inhomogeneous boundary conditions is currently limited to its use in freebc.jl.
=#

abstract type HomogeneousBoundaryConditions; end

struct GeneralHomogeneousBoundaryConditions{D} <: HomogeneousBoundaryConditions
    a::SArray{NTuple{D, 3}, Int8}
end


"""
A boundary condition that just extrapolates the value in all boundaries.  This is used e.g.
for the eabs field.
"""
struct ExtrapolateConst <: HomogeneousBoundaryConditions; end

"""
Builds an efficient boundary condition representation from a list of D
tuples with the boundary condition for the lowest and highest value
of the axis in that dimension.
"""
function boundaryconditions(desc::NTuple{D, Tuple{Int, Int}}) where D
    a = ones(Int8, ntuple(_ -> 3, Val(D))...)
    for (i, (l, r)) in enumerate(desc)
        linds = CartesianIndices(ntuple(d -> (d == i ? (1:1) : (1:3)), Val(D)))
        rinds = CartesianIndices(ntuple(d -> (d == i ? (3:3) : (1:3)), Val(D)))
        a[linds] .*= l
        a[rinds] .*= r        
    end        

    return GeneralHomogeneousBoundaryConditions(SArray{NTuple{D, 3}, Int8}(a))
end

"""
Returns the value of the ghost cell u[n+1] beyond the domain baoundary given the boundary conditions
`bc`, the face (a vector in {0, 1}^D perpendicular to the boundary) and the value u[n] of the cell closest
to the face.
"""
function getbc(bc::GeneralHomogeneousBoundaryConditions{D}, face::CartesianIndex{D}, uclose) where D
    return bc.a[ntuple(d -> 2 + face[d], Val(D))...] * uclose 
end

getbc(::ExtrapolateConst, face, uclose) = uclose
