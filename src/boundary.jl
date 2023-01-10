#=
  Boundary conditions are represented by a 3x3 or 3x3x3 StaticArray.  

  Each element stores +1/-1 for Neumann / Dirchlet b.c.; the center element
  is ignored.
=#


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

    return SArray{NTuple{D, 3}, Int8}(a)
end


function getbc(bc::SArray{NTuple{D, 3}, Int8}, face::CartesianIndex{D}) where D
    return bc[ntuple(d -> 2 + face[d], Val(D))...] 
end
