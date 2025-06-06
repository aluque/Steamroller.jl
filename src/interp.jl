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

"""
Interpolates from a rectangle of an D-dimensional array `src` to another 
D-dimensional array `dest`.

If `base` exists the results is added to the corresponding value there.
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

