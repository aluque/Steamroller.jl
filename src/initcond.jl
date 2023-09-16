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


