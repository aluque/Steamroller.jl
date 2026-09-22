#=
  Interpolation.
=#


# Hard-coded interpolation weights for 2D and 3D
@inline __w0(::Val{2}) = 1 / 2
@inline __w1(::Val{2}) = 1 / 4

@inline __w0(::Val{3}) = 1 / 4
@inline __w1(::Val{3}) = 1 / 4

@inline _first(x, y) = x

struct InterpLinear end;
struct InterpCopy end;
struct InterpLimited end;
"""
This pseudo-quadratic interpolation improves a bit the linear interpolation. However, because it
uses the same stencil as the InterpLinear, it only achieves quadratic accuracy if the interpolated
function is separable along the axes (e.g. f(r, z) = f(r) + f(z)).

However for general functions it improves over linear interpolation by cancelling the error
proportional to `∇²f`; if f is a potential this is proportional to the charge density so we are
removing a systematic error wherever the charge concentrates.

`P = true` clamps the result at zero, for densities.
"""
struct InterpPseudoQuadratic{P} end
InterpPseudoQuadratic() = InterpPseudoQuadratic{false}()


"""
Map an input symbol to an interpolation method.  For InterpPseudoQuadratic, `clamp`
ensures positivity.
"""
function getinterp(sym::Symbol, clamp::Bool=false)
    sym == :limited ? InterpLimited() :
        sym == :linear ? InterpLinear() :
        sym == :pseudoquadratic ? InterpPseudoQuadratic{clamp}() :
        sym == :copy ? InterpCopy() :
        throw(ArgumentError("Unknown interpolation method $sym"))
end

"""
The minmod limiter: zero if `a` and `b` have opposite signs, otherwise the one with the smaller
magnitude.
"""
@inline function minmod(a, b)
    (a * b <= 0) && return zero(a)
    abs(a) < abs(b) ? a : b
end

"""
Interpolate linearly from a rectangle of an D-dimensional array `src` to another 
D-dimensional array `dest`.

`func` allows to combine the interpolation with an previous value in the `dest` field (e.g. `+` adds
the value). Default is to overwrite.
"""
@generated function interp!(::InterpLinear,
                            dest::AbstractArray{T, D}, idest::CartesianIndices{D},
                            src::AbstractArray{T, D}, isrc::CartesianIndices{D},
                            func::B=_first) where {T, D, B}
    quote
        # @assert all(ntuple(d -> (2 * length(isrc.indices[d]) == length(idest.indices[d])),
        #                         Val(D))) "Incompatible sizes in interpolation"
        # Is, Id -> Indexes of source, dest inside the rectangles
        # Js, Jd -> Indexes in the original arrays (e.g. Js = isrc[Is])
        for Is in CartesianIndices(isrc)
            Id0 = 2 * Is - oneunit(Is)
            Jd0 = idest[Id0]
            
            Js = isrc[Is]
            
            # for R in CubeStencil{D, 0}()
            #     # We will compute the value of cell in dest at this location
            #     Jd = Jd0 + R
            
            #     f = __w0(Val(D)) * src[Js]
            #     for i in 1:D
            #         L = CartesianIndex(ntuple(d -> (i == d ? (2 * R[d] - 1) : 0), Val(D)))
            #         f += __w1(Val(D)) * src[Js + L]
            #     end
            #     dest[Jd] = f
            # end
            @nloops $D i d->0:1 begin
                # We will compute the value of cell in dest at this location
                Jd = CartesianIndex(@ntuple $D d -> Jd0[d] + i_d)
                
                f = __w0(Val($D)) * src[Js]
                for j in 1:$D
                    #L = CartesianIndex(ntuple(d -> (i == d ? (2 * R[d] - 1) : 0), Val(D)))
                    f += __w1(Val($D)) * @nref $D src d -> (d == j ? Js[d] + 2 * i_d - 1 : Js[d])
                end
                dest[Jd] = func(f, dest[Jd])
            end
        end
    end
end


"""
Interpolate linearly with slope-limiting from a rectangle of an D-dimensional array `src` to another 
D-dimensional array `dest`.

`func` allows to combine the interpolation with an previous value in the `dest` field (e.g. `+` adds
the value). Default is to overwrite.
"""
@generated function interp!(::InterpLimited,
                            dest::AbstractArray{T, D}, idest::CartesianIndices{D},
                            src::AbstractArray{T, D}, isrc::CartesianIndices{D},
                            func::B=_first) where {T, D, B}
    quote
        # Is, Id -> Indexes of source, dest inside the rectangles
        # Js, Jd -> Indexes in the original arrays (e.g. Js = isrc[Is])
        for Is in CartesianIndices(isrc)
            Id0 = 2 * Is - oneunit(Is)
            Jd0 = idest[Id0]

            Js = isrc[Is]
            u0 = src[Js]

            # Limited slopes scaled by 1/4: the offset of a child center from the coarse center
            # is one quarter of the coarse cell size.
            s = @ntuple $D j -> begin
                up = @nref $D src d -> (d == j ? Js[d] + 1 : Js[d])
                um = @nref $D src d -> (d == j ? Js[d] - 1 : Js[d])
                minmod(up - u0, u0 - um) / 4
            end

            @nloops $D i d->0:1 begin
                Jd = CartesianIndex(@ntuple $D d -> Jd0[d] + i_d)
                sg = @ntuple $D d -> 2 * i_d - 1

                f = u0
                for j in 1:$D
                    f += sg[j] * s[j]
                end
                dest[Jd] = func(f, dest[Jd])
            end
        end
    end
end


"""
Interpolate from a rectangle of an D-dimensional array `src` to another D-dimensional array `dest`
using the direction-wise quadratic fit described in `InterpPseudoQuadratic`.

`func` allows to combine the interpolation with an previous value in the `dest` field (e.g. `+` adds
the value). Default is to overwrite.
"""
@generated function interp!(::InterpPseudoQuadratic{P},
                            dest::AbstractArray{T, D}, idest::CartesianIndices{D},
                            src::AbstractArray{T, D}, isrc::CartesianIndices{D},
                            func::B=_first) where {P, T, D, B}
    quote
        # Is, Id -> Indexes of source, dest inside the rectangles
        # Js, Jd -> Indexes in the original arrays (e.g. Js = isrc[Is])
        for Is in CartesianIndices(isrc)
            Id0 = 2 * Is - oneunit(Is)
            Jd0 = idest[Id0]

            Js = isrc[Is]
            u0 = src[Js]

            # Centered slope (up - um) / 2 per coarse cell, times the quarter-cell offset of the
            # fine cell center.  The curvature of the fitted parabola cancels between the two
            # half cells, so this is exact on cell averages.
            s = @ntuple $D j -> begin
                up = @nref $D src d -> (d == j ? Js[d] + 1 : Js[d])
                um = @nref $D src d -> (d == j ? Js[d] - 1 : Js[d])
                (up - um) / 8
            end

            @nloops $D i d->0:1 begin
                Jd = CartesianIndex(@ntuple $D d -> Jd0[d] + i_d)
                sg = @ntuple $D d -> 2 * i_d - 1

                f = u0
                for j in 1:$D
                    f += sg[j] * s[j]
                end

                f = $(P ? :(max(f, zero(f))) : :f)
                dest[Jd] = func(f, dest[Jd])
            end
        end
    end
end


@generated function interp!(::InterpCopy,
                            dest::AbstractArray{T, D}, idest::CartesianIndices{D},
                            src::AbstractArray{T, D}, isrc::CartesianIndices{D},
                            func::B=_first) where {T, D, B}
    quote
        # Is, Id -> Indexes of source, dest inside the rectangles
        # Js, Jd -> Indexes in the original arrays (e.g. Js = isrc[Is])
        for Is in CartesianIndices(isrc)
            Id0 = 2 * Is - oneunit(Is)
            Jd0 = idest[Id0]
            
            Js = isrc[Is]
            @nloops $D i d->0:1 begin
                # We will compute the value of cell in dest at this location
                Jd = CartesianIndex(@ntuple $D d -> Jd0[d] + i_d)
                dest[Jd] = src[Js]
            end
        end
    end
end


interp!(dest::AbstractArray{T, D}, idest::CartesianIndices{D},
        src::AbstractArray{T, D}, isrc::CartesianIndices{D},
        func::B=_first) where {T, D, B} = interp!(InterpLinear(), dest, idest, src, isrc, func)


"""
Fill blocks at a given level with interpolations from their parent blocks.
"""
function interp_level!(u::ScalarBlockField{D}, v::Vector{Child{D}}) where {D}
    @batch for edge in v
        src = u[edge.coarse]
        dest = u[edge.fine]
        interp!(dest, validindices(u), src, subblockindices(u, edge.subblock))
    end
end

"""
Interpolate from u1 adding the result to u.
"""
function interp_add_level!(u::ScalarBlockField{D}, u1::ScalarBlockField{D},
                           v::Vector{Child{D}}) where {D}
    @batch for edge in v
        src = u1[edge.coarse]
        dest = u[edge.fine]
        
        interp!(dest, validindices(u), src, subblockindices(u1, edge.subblock), +)
    end
end


"""
Fill all children blocks with interpolations from their parents.
"""
function interp_full!(u::ScalarBlockField{D}, conn::Connectivity, bc, func=_first) where {D}
    for i in 2:length(conn.child)
        fill_ghost_copy!(u, conn.neighbor[i - 1])
        fill_ghost_bnd!(u, conn.boundary[i - 1], bc)
        interp_blocks!(u, conn.child[i])
    end
end

