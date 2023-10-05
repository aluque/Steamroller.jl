#=
Some help methods to handle gases with a changing density.

The key is the method nscale, which has to be evaluated at a given location and returns a scaling
factor for eabs, transport parameters etc. We provide a trivial scaling factor that just evaluates to 1
=#

abstract type AbstractDensityScaling; end

struct TrivialDensityScaling <: AbstractDensityScaling; end

nscale(::TrivialDensityScaling, x) = 1

struct ExpDensityScaling{T} <: AbstractDensityScaling;
    "Scale height"
    h::T

    "Reference height"
    z0::T
end

nscale(s::ExpDensityScaling, x) = exp(-(x[end] - s.z0) / s.h)



