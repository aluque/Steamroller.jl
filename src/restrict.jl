"""
Restricts from a rectangle of an D-dimensional array `src` to another 
D-dimensional array `dest`.
"""
@generated function restrict!(dest::AbstractArray{T, D}, idest::CartesianIndices{D},
                              src::AbstractArray{T, D}, isrc::CartesianIndices{D}) where {T, D}
    quote
        k = (1 << $D)

        # Is, Id -> Indexes of source, dest inside the rectangles
        # Js, Jd -> Indexes in the original arrays (e.g. Js = isrc[Is])
        for Id in CartesianIndices(idest)
            Is0 = 2 * Id - oneunit(Id)

            Js0 = isrc[Is0]
            Jd = idest[Id]

            f = zero(T)            
            @nloops $D i d->0:1 begin
                f += @nref $D src d -> Js0[d] + i_d
            end
            (@nref $D dest d -> Jd[d]) = f / k
        end
    end
end


"""
Fill blocks at a given level with restrictions from their parent blocks.
"""
function restrict_level!(u::ScalarBlockField{D}, v::Vector{Child{D}}) where {D}
    @batch for edge in v
        src = u[edge.fine]
        dest = u[edge.coarse]
        restrict!(dest, subblockindices(u, edge.subblock), src, validindices(u))
    end
end


"""
Fill all parent blocks with restrict from their children.
"""
function restrict_full!(u::ScalarBlockField{D}, conn::Connectivity) where {D}
    for i in length(conn.child):-1:1
        restrict_level!(u, conn.child[i])
    end
end


"""
Restrict fluxes from two blocks.  For blocks of dimension `D` this restriction has dimension `D` - 1.
The fixed dimension (perpendicular to the face) is `dim`. `idest` and `isrc` are `CartesianIndices` with
dimension `D` even though one of the dimensions is fixed (singleton).
"""
@generated function restrict_flux!(dest::AbstractArray{T, D1}, idest::CartesianIndices{D},
                                   src::AbstractArray{T, D1}, isrc::CartesianIndices{D},
                                   dim) where {T, D1, D}
    # Remember that the last dimension of dest/src is the vector component
    @assert D1 == D + 1
    quote
        k = (1 << ($D - 1))

        # Is, Id -> Indexes of source, dest inside the rectangles
        # Js, Jd -> Indexes in the original arrays (e.g. Js = isrc[Is])
        for Id in CartesianIndices(idest)
            Js0 = @nref $D isrc d -> (d != dim ? 2 * Id[d] - 1 : Id[d])
            Jd = idest[Id]

            f = zero(T)
            @nloops $D i d->(d == dim ? (0:0) : (0:1)) begin
                # The last index is dim because, remember, dest and src are vector arrays so we need
                # the component index.
                f += @nref $D1 src d -> (d == $D1 ? dim : Js0[d] + i_d)
            end
            (@nref $D1 dest d -> (d == $D1 ? dim : Jd[d])) = f / k
        end
    end
end
