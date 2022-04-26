#=
  We use metaprogramming to compute efficiently and in a general way all 
  laplacians.  The stencils are expanded at compile time and 0s do not cause
  any performance penalty.
=#

# The standard 5-point or 7-point laplacian stencils
#
# For performance reasons the stencils are applied using a @generated function
# and hat means that the stencil values should be obtained from type
# information.
###################################################

"""
    Represent a Laplacian discretization in dimension D and order P.
    Only P=2 and P=4 are supported now.
"""
struct LaplacianDiscretization{D,P}; end

stencil(::LaplacianDiscretization{2, 2}, ::Val{:lhs}, geom) = @SArray [0  1  0;
                                                                       1 -4  1;
                                                                       0  1  0]

denom(::LaplacianDiscretization{2, 2}, ::Val{:lhs}, geom) = 1

stencil(::LaplacianDiscretization{3, 2}, ::Val{:lhs}, geom) = @SArray [[0   0   0;
                                                                        0   1   0;
                                                                        0   0   0];;;
                                                                       
                                                                       [0   1   0;
                                                                        1  -6   1;
                                                                        0   1   0];;;
                                                                       
                                                                       [0   0   0;
                                                                        0   1   0;
                                                                        0   0   0]]

denom(::LaplacianDiscretization{3, 2}, ::Val{:lhs}, geom) = 1

# Thsese discretizations have a trivial RHS stencil
stencil(::LaplacianDiscretization{2, 2}, ::Val{:rhs}, geom) = @SArray [0  0  0;
                                                                       0  1  0;
                                                                       0  0  0]
denom(::LaplacianDiscretization{2, 2}, ::Val{:rhs}, geom) = 1

stencil(::LaplacianDiscretization{3, 2}, ::Val{:rhs}, geom) = @SArray [[0   0   0;
                                                                        0   0   0;
                                                                        0   0   0];;;
                                                                       
                                                                       [0   0   0;
                                                                        0   1   0;
                                                                        0   0   0];;;
                                                                       
                                                                       [0   0   0;
                                                                        0   0   0;
                                                                        0   0   0]]
denom(::LaplacianDiscretization{3, 2}, ::Val{:rhs}, geom) = 1


# The 4th-order Mehrstellen stencil.
# See e.g. Trottenberg, Oosterlee and Schuler, 5.4.2
###################################################

stencil(::LaplacianDiscretization{2, 4}, ::Val{:lhs}, ::CartesianGeometry) = @SArray [1    4  1;
                                                                                      4  -20  4;
                                                                                      1    4  1]
denom(::LaplacianDiscretization{2, 4}, ::Val{:lhs}, ::CartesianGeometry) = 6

stencil(::LaplacianDiscretization{2, 4}, ::Val{:lhs}, ::CylindricalGeometry) = @SArray [1    4  1;
                                                                                        4  -20  4;
                                                                                        1    4  1]
denom(::LaplacianDiscretization{2, 4}, ::Val{:lhs}, ::CylindricalGeometry) = 6


stencil(::LaplacianDiscretization{3, 4}, ::Val{:lhs}, ::CartesianGeometry) = @SArray [[0   1   0;
                                                                                       1   2   1;
                                                                                       0   1   0];;;
                                                                                      
                                                                                      [1    2   1;
                                                                                       2   -24  2;
                                                                                       1    2   1];;;
                                                                                      
                                                                                      [0   1   0;
                                                                                       1   2   1;
                                                                                       0   1   0]]
denom(::LaplacianDiscretization{3, 4}, ::Val{:lhs}, ::CartesianGeometry) = 6

stencil(::LaplacianDiscretization{2, 4}, ::Val{:rhs}, ::CartesianGeometry) = @SArray [0   1  0;
                                                                                      1   8  1;
                                                                                      0   1  0]
denom(::LaplacianDiscretization{2, 4}, ::Val{:rhs}, ::CartesianGeometry) = 12

stencil(::LaplacianDiscretization{2, 4}, ::Val{:rhs}, ::CylindricalGeometry) = @SArray [0   1  0;
                                                                                        1  14  1;
                                                                                        0   1  0]
denom(::LaplacianDiscretization{2, 4}, ::Val{:rhs}, ::CylindricalGeometry) = 18

stencil(::LaplacianDiscretization{3, 4}, ::Val{:rhs}, ::CartesianGeometry) = @SArray [[0   0   0;
                                                                                       0   1   0;
                                                                                       0   0   0];;;
                                                                                      
                                                                                      [0    1   0;
                                                                                       1    6   1;
                                                                                       0    1   0];;;
                                                                                      
                                                                                      [0   0   0;
                                                                                       0   1   0;
                                                                                       0   0   0]]

denom(::LaplacianDiscretization{3, 4}, ::Val{:rhs}, ::CartesianGeometry) = 12

@inline function center_factor(ld::LaplacianDiscretization{D}, side, geom) where {D}
    stencil(ld, side, geom)[2 * oneunit(CartesianIndex{D})]
end

_gridstep(::Val{:lhs}) = 1
_gridstep(::Val{:rhs}) = 2

    
@generated function applystencil(u, I::CartesianIndex{D}, J::CartesianIndex{D},
                      geom::G, ::LaplacianDiscretization{D, P},
                      ::Val{side}) where {D, P, G, side}
    ld = LaplacianDiscretization{D, P}()
    g = G()
    cf = center_factor(ld, Val(side), g)
    k = denom(ld, Val(side), g)

    expr = quote
        $(Expr(:meta, :inline))
        # Start with center value
        v = $cf * u[I]
    end

    gstep = _gridstep(Val(side))
    starray = stencil(ld, Val(side), g)
    
    for S in CartesianIndices(starray)
        f = starray[S]
        f != 0 || continue

        shift = ntuple(d -> gstep * (S[d] - 2), Val(D))

        # The center value has already been included
        all(==(0), shift) && continue
        
        push!(expr.args, :(I1 = I))
        for d in 1:D
            if shift[d] != 0
                push!(expr.args,
                      :(I1 = Base.setindex(I1, I1[$d] + $(shift[d]), $d)))
            end
        end
        #push!(expr.args, :(@show J factor(geom, J, $shift)))
        push!(expr.args, :(v += $f * factor(geom, J, $shift) * u[I1]))
    end
    push!(expr.args, :(return v / $k))

    return expr
end
