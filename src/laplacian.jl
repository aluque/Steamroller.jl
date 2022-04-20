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

stencil(::LaplacianDiscretization{2, 2}, ::Val{:lhs}) = [0  1  0;
                                                         1 -4  1;
                                                         0  1  0]

stencil(::LaplacianDiscretization{3, 2}, ::Val{:lhs}) = [[0   0   0;
                                                          0   1   0;
                                                          0   0   0];;;
                                                         
                                                         [0   1   0;
                                                          1  -6   1;
                                                          0   1   0];;;
                                                         
                                                         [0   0   0;
                                                          0   1   0;
                                                          0   0   0]]

# Thses discretizations do not include any RHS stencil
stencil(::LaplacianDiscretization{2, 2}, ::Val{:rhs}) = [0  0  0;
                                                         0  1  0;
                                                         0  0  0]

stencil(::LaplacianDiscretization{3, 2}, ::Val{:rhs}) = [[0   0   0;
                                                          0   0   0;
                                                          0   0   0];;;
                                                         
                                                         [0   0   0;
                                                          0   1   0;
                                                          0   0   0];;;
                                                         
                                                         [0   0   0;
                                                          0   0   0;
                                                          0   0   0]]



# The 4th-order Mehrstellen stencil.
# See e.g. Trottenberg, Oosterlee and Schuler, 5.4.2
###################################################

stencil(::LaplacianDiscretization{2, 4}, ::Val{:lhs}) = [1    4  1;
                                                         4  -20  4;
                                                         1    4  1]

stencil(::LaplacianDiscretization{3, 4}, ::Val{:lhs}) = [[0   1   0;
                                                          1   2   1;
                                                          0   1   0];;;
                                                         
                                                         [1    2   1;
                                                          2   -24  2;
                                                          1    2   1];;;
                                                         
                                                         [0   1   0;
                                                          1   2   1;
                                                          0   1   0]]

stencil(::LaplacianDiscretization{2, 4}, ::Val{:rhs}) = [0   1  0;
                                                         1   8  1;
                                                         0   1  0]

stencil(::LaplacianDiscretization{3, 4}, ::Val{:rhs}) = [[0   0   0;
                                                          0   1   0;
                                                          0   0   0];;;
                                                         
                                                         [0    1   0;
                                                          1    6   1;
                                                          0    1   0];;;
                                                         
                                                         [0   0   0;
                                                          0   1   0;
                                                          0   0   0]]

@inline function center_factor(ld::LaplacianDiscretization{D}, side) where {D}
    stencil(ld, side)[2 * oneunit(CartesianIndex{D})]
end


"""
    Receives a `stencil` as an array and a ref expression `refexpr` 
    (e.g. `:(A[i, j])`) and produces an expression evaluating the stencil 
    around `A[i, j]`.  geomsym is a symbol to call the geometry factor.
"""
function stencilexpr(stencil::AbstractArray, refexpr::Expr,
                     globalcoords, geom::AbstractGeometry)
    @assert refexpr.head == :ref

    arr = refexpr.args[1]
    z = refexpr.args[2:end]
    
    exprs = Expr[]
    D = length(z)
    
    for I in CartesianIndices(stencil)
        f = stencil[I]
        f != 0 || continue

        gf = factorexpr(geom, globalcoords, I - 2 * oneunit(I))
        
        inds = ntuple(d -> :($(I[d] - 2) + $(z[d])), D)
        ref = Expr(:ref, arr, inds...)
        
        push!(exprs, :($f * $gf * $ref))
    end
    
    Expr(:call, :+, exprs...)
end


@generated function applystencil(u, I::CartesianIndex{D}, J::CartesianIndex{D},
                      geom, ::LaplacianDiscretization{D, P},
                      ::Val{side}) where {D, P, side}
    ld = LaplacianDiscretization{D, P}()
    cf = center_factor(ld, Val(side))
    
    expr = quote
        $(Expr(:meta, :inline))
        # Start with center value
        v = $cf * u[I]
    end

    starray = stencil(ld, Val(side))
    
    for S in CartesianIndices(starray)
        f = starray[S]
        f != 0 || continue

        shift = ntuple(d -> S[d] - 2, Val(D))

        # The center value has already been included
        all(==(0), shift) && continue
        
        push!(expr.args, :(I1 = I))
        for d in 1:D
            if shift[d] != 0
                push!(expr.args,
                      :(I1 = Base.setindex(I1, I1[$d] + $(shift[d]), $d)))
            end
        end
        push!(expr.args, :(v = muladd($f * factor(geom, J, $shift),  u[I1], v)))
    end
    push!(expr.args, :(return v))

    return expr
end
