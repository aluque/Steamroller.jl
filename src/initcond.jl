#=
A library of initial conditions.
=#
abstract type AbstractInitialCondition{T} end

"""
An uniform background.
"""
@kwdef struct Background{T} <: AbstractInitialCondition{T}
    "Background level"
    bg::T
end

(f::Background)(args...) = f.bg
    

"""
A (Super-)Gaussian initial condition.
"""
@kwdef struct Gaussian{T} <: AbstractInitialCondition{T}
    "Amplitude factor"
    A::T

    "e-folding width"
    w::T

    "x coordinate of the center"
    x0::T = zero(A)

    "y coordinate of the center"
    y0::T = zero(A)

    "x coordinate of the center"
    z0::T = zero(A)

    "Exponent of the super-gaussian"
    s::Int = 1
    
    "if nonzero, extend the gaussian in the z axis in the given direction"
    extend::Int = 0
end

_extend(z, dir) = dir == 0 ? z : -dir * max(0, -dir * z)

function (f::Gaussian)(x, y, z)
    (;A, w, x0, y0, z0, s, extend) = f
    return A * exp(-(((x - x0)^2 + (y - y0)^2 + _extend(z - z0, extend)^2) / w^2)^s)
end

function (f::Gaussian)(r, z)
    (;A, w, x0, y0, z0, s, extend) = f
    if (x0 != 0 || y0 != 0)
        @warn "x0 [=$x0] and y0 [=$y0] ignored in cylindrically symmetrical gaussian"
    end

    return A * exp(-((r^2 + _extend(z - z0, f.extend)^2) / w^2)^s)
end


@kwdef struct Cylinder{T, F} <: AbstractInitialCondition{T}
    "Function to set density at z"
    func::F
    
    "Radius of the cylinder"
    R::T

    "x coordinate of the center"
    x0::T = zero(R)

    "y coordinate of the center"
    y0::T = zero(R)

    "Base of the cylinder"
    z0::T = typemin(R)

    "Top of the cylinder"
    z1::T = typemax(R)
end


function (f::Cylinder)(x, y, z)
    (;func, R, x0, y0, z0, z1) = f
    return ifelse(z0 < z < z1 && (x - x0)^2 + (y - y0)^2 < R^2, func(z), zero(R)) 
end

function (f::Cylinder)(r, z)
    (;func, R, z0, z1) = f
    return ifelse(z0 < z < z1 && r < R, func(z), zero(R))
end


"""
A funnel-like density where we vary the radius and density at the center.
"""
@kwdef struct Funnel{T, FR, FN} <: AbstractInitialCondition{T}
    "Function to set density at z"
    n0::FN
    
    "Radius of the cylinder"
    a::FR

    "Normalizing radius"
    a0::T
    
    "x coordinate of the center"
    x0::T = zero(a0)

    "y coordinate of the center"
    y0::T = zero(a0)

    "Base of the cylinder"
    z0::T = typemin(a0)

    "Top of the cylinder"
    z1::T = typemax(a0)
end


function (f::Funnel{T})(x, y, z) where T
    (;n0, a, a0, x0, y0, z0, z1) = f
    r = sqrt((x - x0)^2 + (y - y0)^2)
    if z0 < z < z1 && r < a(z)
        return n0(z) * (a(z) / a0)^2 * max(zero(T), 1 - (r / a(z))^2)
    else
        return zero(T)
    end    
end

function (f::Funnel{T})(r, z) where T
    (;n0, a, a0, z0, z1) = f
    if z0 < z < z1 && r < a(z)
        return n0(z) * (a0 / a(z))^2 * max(zero(T), 1 - (r / a(z))^2)
    else
        return zero(T)
    end    
end


"""
The addition of several initial conditions.
"""
struct SumInitialCondition{T, TPL <: Tuple{Vararg{<:AbstractInitialCondition{T}}}} <: AbstractInitialCondition{T}
    ic::TPL
end

@generated function (f::SumInitialCondition{T, TPL})(args...) where {T, TPL}
    L = fieldcount(TPL)
    out = quote
        v = zero($T)
    end
    
    for i in 1:L
        push!(out.args,
              quote
              v += f.ic[$i](args...)
              end
              )
    end
    push!(out.args, :(return v))
    
    out    
end

Base.:+(args::AbstractInitialCondition{T}...) where {T} = SumInitialCondition(args)


