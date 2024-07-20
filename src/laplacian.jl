#=
Discretizations of the Laplacian operator.

Note: All functions here receive a parameter s that is ignored. This is needed only for the
Helmholtz discretizations in helmholtz.jl but must be included here share the same code in the
multigrid solver.
=#

abstract type AbstractDiscretization{D, P}; end
"""
Represent a Laplacian discretization in dimension D and order P.
Only P=2 and P=4 are supported now.
"""
struct LaplacianDiscretization{D,P} <: AbstractDiscretization{D, P}; end


"""
Build an expression that applies a static stencil.

`expr1` is a reference espression such as `u[I]`, `expr2` is the stencil
provided in place (not as a variable).  Optionally the full stencil 
application is divided by `denom`, which allows us to avoid too many
floating point operations.
""" 
macro stencil(expr1, expr2, denom=1)
    @capture(expr1, u_[I_]) ||
        error("First expression must be a reference to an array")
    
    arr = _exprarray(expr2)
    D = length(size(arr))
    expr = quote v = zero(eltype($(esc(u)))) end
    
    for S1 in CartesianIndices(arr)
        arr[S1] != 0 || continue
        S = S1 - 2 * oneunit(S1)
        
        push!(expr.args, :(I1 = $(esc(I))))
        for d in 1:D
            if S[d] != 0
                push!(expr.args,
                      :(I1 = Base.setindex(I1, I1[$d] + $(S[d]), $d)))
            end
        end
        push!(expr.args, :(v += $(esc(arr[S1])) * $(esc(u))[I1]))
    end
    push!(expr.args, :(v / $(esc(denom))))

    return expr
end


function _exprarray(expr)
    # 3D stencil
    if @capture(expr, [a_;;; b_;;; c_])
        return [_exprarray(a);;; _exprarray(b);;; _exprarray(c)]
    end

    if @capture(expr, [a_; b_; c_])
        return [_exprarray(a); _exprarray(b); _exprarray(c)]
    end

    if @capture(expr, a_row)        
        r = Array{Any, 2}(undef, 1, 3)
        r[1, :] = a.args
        return r
    end    
end

#= 
Note: All the stencils defined here ignore the input `s`, which is only required for helmholtz
discretizations.
=#

##
#  2D STENCILS
##

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 2},
                              ::CartesianGeometry,
                              ::Val{:lhs})
    @inbounds @fastmath @stencil u[I] [0  1  0;
                   1 -4  1;
                   0  1  0]
end

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 2},
                              ::CartesianGeometry,
                              ::Val{:rhs})
    @inbounds @fastmath u[I]
end

@inline diagelm(s, ::LaplacianDiscretization{2, 2},
                    ::CartesianGeometry, ::Val{:lhs}, J) = -4


## CYLINDRICAL

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 2},
                              ::CylindricalGeometry{1},
                              ::Val{:lhs})
    i = J[1]
    @inbounds @fastmath @stencil u[I] [0  (2i - 2) / (2i - 1)  0;
                   1                   -4  1;
                   0         2i / (2i - 1) 0]
end

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 2},
                              ::CylindricalGeometry{2},
                              ::Val{:lhs})
    i = J[2]
    @inbounds @fastmath @stencil u[I] [0                      1                0;
                                       ((2i - 2) / (2i - 1)) -4  (2i / (2i - 1));
                                       0                      1                0]
end

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 2},
                              ::CylindricalGeometry,
                              ::Val{:rhs})
    @inbounds u[I]
end

@inline diagelm(s, ::LaplacianDiscretization{2, 2},
                ::CylindricalGeometry, ::Val{:lhs}, J) = -4


# COMPACT 4TH ORDER 2D CARTESIAN

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 4},
                              ::CartesianGeometry,
                              ::Val{:lhs})
    @inbounds @fastmath @stencil u[I] [1    4  1;
                                       4  -20  4;
                                       1    4  1] 6
end

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 4},
                              ::CartesianGeometry,
                              ::Val{:rhs})
    @inbounds @fastmath @stencil u[I] [0   1  0;
                                       1   8  1;
                                       0   1  0] 12
end

@inline diagelm(s, ::LaplacianDiscretization{2, 4},
                    ::CartesianGeometry, ::Val{:lhs}, J) = -20 / 6


# COMPACT 4TH ORDER 2D CYLINDRICAL

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 4},
                              ::CylindricalGeometry{1},
                              ::Val{:lhs})
    i = J[1]
    @inbounds @fastmath @stencil u[I] [2*(i-1)/(2i-1)    4*(-2+9i-16i^2+8i^3)/(2i-1)^3  2*(i-1)/(2i-1);
                                       4                      -16*(1-5i+5i^2)/(1-2i)^2               4;
                                       2i/(2i-1)            4*(1+i-8i^2+8i^3)/(2i-1)^3        2i/(2i-1)] 6
end

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 4},
                              ::CylindricalGeometry{1},
                              ::Val{:rhs})
    i = J[1]
    @inbounds @fastmath @stencil u[I] [0   (2i-2)/(2i-1)  0;
                                       1               8  1;
                                       0       2i/(2i-1)  0] 12
end

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 4},
                              ::CylindricalGeometry{2},
                              ::Val{:lhs})
    i = J[2]
    @inbounds @fastmath @stencil u[I] [2*(i-1)/(2i-1)                                        4   2i/(2i-1);
                                       4*evalpoly(i, (-2, 9, -16, 8))/(2i-1)^3  -16*evalpoly(i, (1, -5, 5))/(1-2i)^2 4*evalpoly(i, (1, 1,-8, 8))/(2i-1)^3;
                                       2*(i-1)/(2i-1)                                        4          2i/(2i-1)] 6
end

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::LaplacianDiscretization{2, 4},
                              ::CylindricalGeometry{2},
                              ::Val{:rhs})
    i = J[2]
    @inbounds @fastmath @stencil u[I] [0                           1          0;
                                       (2i-2)/(2i-1)               8  2i/(2i-1);
                                       0                           1          0] 12
end


@inline function diagelm(s, ::LaplacianDiscretization{2, 4},
                         ::CylindricalGeometry{N}, ::Val{:lhs}, J) where N
    i = J[N]
    @inbounds @fastmath -16*(1-5i+5i^2)/(1-2i)^2 / 6
end

# @inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
#                               ::LaplacianDiscretization{2, 4},
#                               ::CylindricalGeometry{2},
#                               ::Val{:lhs})
#     i = J[2]
#     @stencil u[I] [2*(i-1)/(2i-1)    4   2i/(2i-1);
#                    8*(i-1)/(2i-1)  -20   8i/(2i-1);
#                    2*(i-1)/(2i-1)    4   2i/(2i-1)] 6
# end

# @inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
#                               ::LaplacianDiscretization{2, 4},
#                               ::CylindricalGeometry{2},
#                               ::Val{:rhs})
#     i = J[2]
#     @stencil u[I] [0                           1                 0;
#                    abs(2i-3)/(2i-1)           14      (2i+1)/(2i-1);
#                    0                           1                 0] 18
# end


# @inline function diagelm(s, ::LaplacianDiscretization{2, 4},
#                          ::CylindricalGeometry{N}, ::Val{:lhs}, J) where N
#     return -20 / 6
# end



# @inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
#                               ::LaplacianDiscretization{2, 4},
#                               ::CylindricalGeometry{2},
#                               ::Val{:lhs})
#     i = J[2]
#     a1 = evalpoly(i, (-28, 2, 48, -24)) / evalpoly(i, (13, -14, -36, 24))
#     a2 = evalpoly(i, (2, 26, 24, -24)) / evalpoly(i, (13, -14, -36, 24))
#     b = -4
#     b1 = -4 * evalpoly(i, (25, -5, -48, 24)) / evalpoly(i, (13, -14, -36, 24))
#     b2 = -4 * evalpoly(i, (4, -29, -24, 24)) / evalpoly(i, (13, -14, -36, 24))
#     c = evalpoly(i, (272, 240, -240)) / evalpoly(i, (13, 12, -12))
#     -(@stencil u[I] [a1  b a2;
#                      b1  c b2;
#                      a1  b a2] 6)
# end

# @inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
#                               ::LaplacianDiscretization{2, 4},
#                               ::CylindricalGeometry{2},
#                               ::Val{:rhs})
#     i = J[2]
#     d = 1
#     q = evalpoly(i, (-17, -12, 12)) / evalpoly(i, (13, -14, -36, 24))
#     f = 8

#     @stencil u[I] [0      d  0;
#                    (d-q)  f  (d+q);
#                    0      d  0] 12
# end


# @inline function diagelm(s, ::LaplacianDiscretization{2, 4},
#                          ::CylindricalGeometry{N}, ::Val{:lhs}, J) where N
#     i = J[N]
#     c = evalpoly(i, (272, 240, -240)) / evalpoly(i, (13, 12, -12))
#     return -c
# end





##
#  3D STENCILS
##

@inline function applystencil(u, s, I::CartesianIndex{3}, J::CartesianIndex{3},
                              ::LaplacianDiscretization{3, 2},
                              ::CartesianGeometry,
                              ::Val{:lhs})
    @inbounds @fastmath @stencil u[I] [[0   0   0;
                                        0   1   0;
                                        0   0   0];;;
                                       
                                       [0   1   0;
                                        1  -6   1;
                                        0   1   0];;;
                                       
                                       [0   0   0;
                                        0   1   0;
                                        0   0   0]]
end

@inline function applystencil(u, s, I::CartesianIndex{3}, J::CartesianIndex{3},
                              ::LaplacianDiscretization{3, 2},
                              ::CartesianGeometry,
                              ::Val{:rhs})
    @inbounds u[I]
end

@inline diagelm(s, ::LaplacianDiscretization{3, 2},
                    ::CartesianGeometry, ::Val{:lhs}, J) = -6

@inline function applystencil(u, s, I::CartesianIndex{3}, J::CartesianIndex{3},
                              ::LaplacianDiscretization{3, 4},
                              ::CartesianGeometry,
                              ::Val{:lhs})
    @inbounds @fastmath @stencil u[I] [[0   1   0;
                                        1   2   1;
                                        0   1   0];;;
                                       
                                       [1    2   1;
                                        2   -24  2;
                                        1    2   1];;;
                                       
                                       [0   1   0;
                                        1   2   1;
                                        0   1   0]] 6
end

@inline function applystencil(u, s, I::CartesianIndex{3}, J::CartesianIndex{3},
                              ::LaplacianDiscretization{3, 4},
                              ::CartesianGeometry,
                              ::Val{:rhs})
    @inbounds @fastmath @stencil u[I] [[0   0   0;
                                        0   1   0;
                                        0   0   0];;;
                                       
                                       [0    1   0;
                                        1    6   1;
                                        0    1   0];;;
                                       
                                       [0   0   0;
                                        0   1   0;
                                        0   0   0]] 12
end

@inline diagelm(s, ::LaplacianDiscretization{3, 4},
                ::CartesianGeometry, ::Val{:lhs}, J) = -4
