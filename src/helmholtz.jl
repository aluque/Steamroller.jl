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

@propagate_inbounds function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              helm::HelmholtzDiscretization{2, 2},
                              ::CartesianGeometry,
                              ::Val{:lhs})
    (;k2) = helm
    @fastmath @stencil u[I] [0         1  0;
                             1 (-4 - s * k2)  1;
                             0         1  0]
end

@propagate_inbounds function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              ::HelmholtzDiscretization{2, 2},
                              ::CartesianGeometry,
                              ::Val{:rhs})
    u[I]
end

@inline diagelm(s, helm::HelmholtzDiscretization{2, 2},
                ::CartesianGeometry, ::Val{:lhs}, J) = -4 - s * helm.k2

@propagate_inbounds function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              helm::HelmholtzDiscretization{2, 2},
                              geom::CylindricalGeometry{1},
                              ::Val{:lhs})
    (cm, cp) = radialcoeffs(geom, J[1])
    (;k2) = helm
    @fastmath @stencil u[I] [0             cm  0;
                             1  (-4 - s * k2)  1;
                             0             cp  0]
end

@propagate_inbounds function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
                              helm::HelmholtzDiscretization{2, 2},
                              geom::CylindricalGeometry{2},
                              ::Val{:lhs})
    (cm, cp) = radialcoeffs(geom, J[2])
    (;k2) = helm
    @fastmath @stencil u[I] [0              1  0;
                             cm (-4 - s * k2)  cp;
                             0              1  0]
end

@propagate_inbounds function applystencil(u, s, I::CartesianIndex{2}, J::CartesianIndex{2},
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

@propagate_inbounds function applystencil(u, s, I::CartesianIndex{3}, J::CartesianIndex{3},
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

@propagate_inbounds function applystencil(u, s, I::CartesianIndex{3}, J::CartesianIndex{3},
                              ::HelmholtzDiscretization{3, 2},
                              ::CartesianGeometry,
                              ::Val{:rhs})
    u[I]
end

@inline diagelm(s, helm::HelmholtzDiscretization{3, 2},
                ::CartesianGeometry, ::Val{:lhs}, J) = -6 - s * helm.k2
