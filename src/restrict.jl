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
function restrict_full!(u::ScalarBlockField{D}, conn::Connectivity, bc) where {D}
    for i in length(conn.child):-1:1
        restrict_blocks!(u, conn.child[i])
    end
end
