#=
Stencils for the Helmholtz discretization.
=#

"""
A Helmholtz discretization in dimension D and order P (only P=2 and P=4 supported)
See also `LaplacianDiscretization{D, P}`.
"""
struct HelmholtzDiscretization{D, P, T} <: AbstractDiscretization{D, P}
    k2::T
end

##
#  2D STENCILS
##

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              helm::HelmholtzDiscretization{2, 2},
                              ::CartesianGeometry,
                              ::Val{:lhs})
    (;k2) = helm
    @fastmath @stencil u[I] [0         1  0;
                             1 (-4 - s * k2)  1;
                             0         1  0]
end

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::HelmholtzDiscretization{2, 2},
                              ::CartesianGeometry,
                              ::Val{:rhs})
    u[I]
end

@inline diagelm(s, helm::HelmholtzDiscretization{2, 2},
                ::CartesianGeometry, ::Val{:lhs}, J) = -4 - s * helm.k2

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              helm::HelmholtzDiscretization{2, 2},
                              ::CylindricalGeometry{1},
                              ::Val{:lhs})
    i = J[1]
    (;k2) = helm
    @fastmath @stencil u[I] [0      (2i - 2) / (2i - 1)  0;
                             1            (-4 - s * k2)  1;
                             0            2i / (2i - 1)  0]
end

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              helm::HelmholtzDiscretization{2, 2},
                              ::CylindricalGeometry{2},
                              ::Val{:lhs})
    i = J[2]
    (;k2) = helm
    @fastmath @stencil u[I] [0                                 1                0;
                             ((2i - 2) / (2i - 1)) (-4 - s * k2)  (2i / (2i - 1));
                             0                                 1                0]
end

@inline function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::HelmholtzDiscretization{2, 2},
                              ::CylindricalGeometry,
                              ::Val{:rhs})
    u[I]
end

@inline diagelm(s, helm::HelmholtzDiscretization{2, 2},
                ::CylindricalGeometry, ::Val{:lhs}, J) = -4 - s * helm.k2


##
#  3D STENCILS
##

@inline function applystencil(u, s, I::CartesianIndex{3}, J::CartesianIndex{3},
                              helm::HelmholtzDiscretization{3, 2},
                              ::CartesianGeometry,
                              ::Val{:lhs})
    (;k2) = helm
    @fastmath @stencil u[I] [[0   0   0;
                              0   1   0;
                              0   0   0];;;
                             
                             [0              1   0;
                              1  (-6 - s * k2)   1;
                              0              1   0];;;
                             
                             [0   0   0;
                              0   1   0;
                              0   0   0]]
end

@inline function applystencil(u, s, I::CartesianIndex{3}, J::CartesianIndex{3},
                              ::HelmholtzDiscretization{3, 2},
                              ::CartesianGeometry,
                              ::Val{:rhs})
    u[I]
end

@inline diagelm(s, helm::HelmholtzDiscretization{3, 2},
                ::CartesianGeometry, ::Val{:lhs}, J) = -6 - s * helm.k2
