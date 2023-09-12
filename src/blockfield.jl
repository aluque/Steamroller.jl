#=
   Routines to handle fields stored in block decks.
=#

abstract type AbstractBlockField{D, M, G, T, N, A}; end


"""
A block field stored in a series of MArrays
Encoded in a type for performance reasons.
`D`: Number of dimensions
`M`: Side length in grid points
`G`: Number of ghost cells
`A`: Container type
"""
struct ScalarBlockField{D, M, G, T, N, A} <: AbstractBlockField{D, M, G, T, N, A}#<: AbstractVectorOfArray{T, N, A}
    #= There are two implementation options for u:
    1. A Vector of MArrays (vector),
    2. A contiguous array (contiguous).

    After testing, it looks that:
    For 2D without optimizations and with bound-checking (vector) is slightly faster but that is 
    reversed with -O3 --bounds-checking=no. The differences are maybe around 10% in both cases.
    For 3D (vector) takes forever to compile (above 30min) but then is significantly faster than 
    (contiguous), both with and without optimizations.  So the time spent in compiling may be worth it
    for simulations that take many hours or even days.

    What to do? For development and in most cases it is better to use (contiguous) however for long
    simulations after debugging, it may be worth to change to (vector).
    =#
    u::A

    function ScalarBlockField{D, M, G, T}(storage::Val{:contiguous}=Val{:contiguous}()) where {D, M, G, T}
        #@assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G
        N = D + 1
        val = ElasticArray{T}(undef, ntuple(_ -> S, Val(D))..., 0)
        new{D, M, G, T, N, typeof(val)}(val)
    end

    function ScalarBlockField{D, M, G, T}(storage::Val{:vector}) where {D, M, G, T}
        #@assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G
        N = D + 1
        val = MArray{NTuple{D, S}, T, D, S^D}[]
        new{D, M, G, T, N, typeof(val)}(val)
    end
    
    function ScalarBlockField{D, M, G, T}(len, storage::Val{:contiguous}=Val{:contiguous}()) where {D, M, G, T}
        @assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G
        N = D + 1
        val = ElasticArray{T}(undef, ntuple(_ -> S, Val(D))..., len)
        val .= 0
        new{D, M, G, T, N, typeof(val)}(val)
    end

    function ScalarBlockField{D, M, G, T}(len, storage::Val{:vector}) where {D, M, G, T}
        @assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G
        N = D + 1
        val = [zero(MArray{NTuple{D, S}, T}) for i in 1:len]
        new{D, M, G, T, N, typeof(val)}(val)
    end
    
    function ScalarBlockField{D, M, G, T, N, A}(u::A) where {D, M, G, T, N, A}
        new{D, M, G, T, N, A}(u)
    end

end

"""
A vector block field encoded in a series of MArrays.

Each MArray corresponding to a block has dimensions
(M + 2G + 1, M + 2G + 1, ..., D).
Usually we use this to store a vector field evaluated at cell
interfaces and related to a gradient of a scalar field.    
"""
struct VectorBlockField{D, M, G, T, N, A} <: AbstractBlockField{D, M, G, T, N, A} #<: AbstractVectorOfArray{T, N, A}
    # We cannot construct the type here so it must be parametric
    u::A

    function VectorBlockField{D, M, G, T}(storage::Val{:contiguous}=Val{:contiguous}()) where {D, M, G, T}
        @assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G + 1
        N = D + 2
        val = ElasticArray{T}(undef, ntuple(_ -> S, Val(D))..., D, 0)
        new{D, M, G, T, N, typeof(val)}(val)
    end

    function VectorBlockField{D, M, G, T}(storage::Val{:vector}) where {D, M, G, T}
        @assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G + 1
        MT = Tuple{ntuple(_ -> S, Val(D))..., D}
        N = D + 2
        val = MArray{MT, T, D + 1, D * S^D}[]
        new{D, M, G, T, N, typeof(val)}(val)
    end
    
    function VectorBlockField{D, M, G, T}(len, storage::Val{:contiguous}=Val{:contiguous}()) where {D, M, G, T}
        @assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G + 1
        N = D + 2
        val = ElasticArray{T}(undef, ntuple(_ -> S, Val(D))..., D, len)
        val .= 0
        new{D, M, G, T, N, typeof(val)}(val)
    end

    function VectorBlockField{D, M, G, T}(len, storage::Val{:vector}) where {D, M, G, T}
        @assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G + 1
        MT = Tuple{ntuple(_ -> S, Val(D))..., D}
        N = D + 2
        val = zeros(MArray{MT, T, D + 1, D * S^D}, len)
        new{D, M, G, T, N, typeof(val)}(val)
    end
    
    function VectorBlockField{D, M, G, T, N, A}(u::A) where {D, M, G, T, N, A}
        new{D, M, G, T, N, A}(u)
    end

end

# Aliases to avoid too much typing
const _ScalarBlockFieldV{D, M, G, T, N, A} = ScalarBlockField{D, M, G, T, N, A} where {A <: Vector{<:MArray}}
const _ScalarBlockFieldA{D, M, G, T, N, A} = ScalarBlockField{D, M, G, T, N, A} where {A <: ElasticArray}
const _VectorBlockFieldV{D, M, G, T, N, A} = VectorBlockField{D, M, G, T, N, A} where {A <: Vector{<:MArray}}
const _VectorBlockFieldA{D, M, G, T, N, A} = VectorBlockField{D, M, G, T, N, A} where {A <: ElasticArray}


_length(u::ElasticArray) = size(u, ndims(u))
_length(u::Vector{<:AbstractArray}) = length(u)
Base.length(f::AbstractBlockField) = _length(f.u)

@propagate_inbounds Base.getindex(f::_ScalarBlockFieldA, c::CartesianIndex, i::Integer) = f.u[c, i]
@propagate_inbounds Base.getindex(f::_VectorBlockFieldA, c::CartesianIndex, d::Integer, i::Integer) = f.u[c, d, i]
@propagate_inbounds Base.setindex!(f::_ScalarBlockFieldA, v, c::CartesianIndex, i::Integer) = f.u[c, i] = v
@propagate_inbounds Base.setindex!(f::_VectorBlockFieldA, v, c::CartesianIndex, d::Integer, i::Integer) = f.u[c, d, i] = v

@propagate_inbounds Base.getindex(f::_ScalarBlockFieldV, c::CartesianIndex, i::Integer) = f.u[i][c]
@propagate_inbounds Base.getindex(f::_VectorBlockFieldV, c::CartesianIndex, d::Integer, i::Integer) = f.u[i][c, d]
@propagate_inbounds Base.setindex!(f::_ScalarBlockFieldV, v, c::CartesianIndex, i::Integer) = f.u[i][c] = v
@propagate_inbounds Base.setindex!(f::_VectorBlockFieldV, v, c::CartesianIndex, d::Integer, i::Integer) = f.u[i][c, d] = v

Base.ndims(f::Type{<:ScalarBlockField{D}}) where D = D + 1

Base.ndims(f::Type{<:VectorBlockField{D}}) where D = D + 2

Base.size(f::Union{_ScalarBlockFieldV, _VectorBlockFieldV}) = size(f.u)

Base.size(f::Union{_ScalarBlockFieldA, _VectorBlockFieldA}) = (blksizeghost(f)..., length(f.u))

Base.copy(f::AbstractBlockField) = typeof(f)(copy(f.u))

@inline function Base.similar(f::_ScalarBlockFieldA{D, M, G}, ::Type{T} = eltype(f)) where {D, M, G, T}
    ScalarBlockField{D, M, G, T}(similar(f.u, T))
end

@inline function Base.similar(f::_VectorBlockFieldA{D, M, G}, ::Type{T} = eltype(f)) where {D, M, G, T}
    VectorBlockField{D, M, G, T}(similar(f.u, T))
end

@inline function Base.similar(f::_ScalarBlockFieldV{D, M, G}, ::Type{T} = eltype(f)) where {D, M, G, T}
    ScalarBlockField{D, M, G, T}([similar(f[i], T) for i in eachindex(f.u)])
end

@inline function Base.similar(f::_VectorBlockFieldV{D, M, G}, ::Type{T} = eltype(f)) where {D, M, G, T}
    VectorBlockField{D, M, G, T}([similar(f[i], T) for i in eachindex(f.u)])
end

Base.axes(f::Union{_ScalarBlockFieldA, _VectorBlockFieldA}) = axes(f.u)
Base.axes(f::Union{_ScalarBlockFieldV, _VectorBlockFieldV}) = (axes(f.u[1])..., axes(f.u)...)
#Base.BroadcastStyle(::Type{<:AbstractBlockField{D, M, G}}) where {D, M, G} = Broadcast.Style{ScalarBlockField{D, M, G}}()
Base.BroadcastStyle(::Type{T}) where {T <: AbstractBlockField} = Broadcast.Style{T}()
Base.BroadcastStyle(::Broadcast.Style{T}, ::T2) where {T <: AbstractBlockField, T2 <: Broadcast.BroadcastStyle} = Broadcast.Style{T}()

function Base.similar(bc::Broadcast.Style{ScalarBlockField{D, M, G}}, ::Type{T}) where {D, M, G, T}
    A = find_abc(bc)
    similar(A, T)
end

function Base.copyto!(f::AbstractBlockField,
                      bc::Broadcast.Broadcasted{<:Broadcast.Style{T}}) where {T <: AbstractBlockField}
    bc = Broadcast.flatten(bc)
    T1 = typeof(Broadcast.BroadcastStyle(eltype(f.u)))
    @batch for i in axes(f.u, ndims(f.u))
        Base.copyto!(f[i], mapbroadcast(T1, i, bc))
    end
    f
end

function Base.copyto!(f::AbstractBlockField,
                      bc::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{N}}) where N
    @batch for i in axes(f.u, ndims(f.u))
        Base.copyto!(f[i], bc)
    end
    f
end
Base.broadcastable(f::AbstractBlockField) = f


"`A = find_abc(As)` returns the first AbstractBlockField among the arguments."
find_abc(bc::Base.Broadcast.Broadcasted) = find_abc(bc.args)
find_abc(args::Tuple) = find_aac(find_abc(args[1]), Base.tail(args))
find_abc(x) = x
find_abc(::Tuple{}) = nothing
find_abc(a::AbstractBlockField, rest) = a
find_abc(::Any, rest) = find_abc(rest)

@propagate_inbounds function Base.getindex(f::_ScalarBlockFieldA{D, M, G, T}, i::Integer) where {D, M, G, T}
    S = M + 2G
    #return SizedArray{NTuple{D, S}, T}(view(f.u, ntuple(_ -> Colon(), Val(D))..., i))
    return view(f.u, ntuple(_ -> Colon(), Val(D))..., i)
end

@propagate_inbounds function Base.getindex(f::_VectorBlockFieldA{D, M, G, T}, i::Integer) where {D, M, G, T}
    S = M + 2G + 1
    MT = Tuple{ntuple(_ -> S, Val(D))..., D}
    #return SizedArray{MT, T}(view(f.u, ntuple(_ -> Colon(), Val(D + 1))..., i))
    return view(f.u, ntuple(_ -> Colon(), Val(D + 1))..., i)
end

@propagate_inbounds function Base.getindex(f::_ScalarBlockFieldV{D, M, G, T}, i::Integer) where {D, M, G, T}
    return f.u[i]
end

@propagate_inbounds function Base.getindex(f::_VectorBlockFieldV{D, M, G, T}, i::Integer) where {D, M, G, T}
    return f.u[i]
end

getblk(f::AbstractBlockField, blk) = f[blk]
valid(f::ScalarBlockField, blk) = view(f[blk], validindices(f))
function valid(f::VectorBlockField{D}, blk, dim) where D
    v = view(f[blk], validindices(f, dim).indices..., dim)
    return v
end


"""
Creates a new block and returns its index.
"""
function newblock!(f::_ScalarBlockFieldA{D, M, G, T}) where {D, M, G, T}
    S = M + 2G
    append!(f.u, Iterators.repeated(zero(T), S^D))
    return length(f)
end

function newblock!(f::_ScalarBlockFieldV{D, M, G, T}) where {D, M, G, T}
    S = M + 2G
    z = zero(MArray{NTuple{D, S}, T})
    push!(f.u, z)
    return length(f)
end

"""
Creates a series of new blocks returns the final length.
"""
function newblocks!(f::_ScalarBlockFieldA{D, M, G, T}, n) where {D, M, G, T}
    S = M + 2G
    append!(f.u, Iterators.repeated(zero(T), n * S^D))
    return length(f)
end

function newblocks!(f::_ScalarBlockFieldV{D, M, G, T}, n) where {D, M, G, T}
    S = M + 2G
    for i in 1:n
        z = zero(MArray{NTuple{D, S}, T})
        push!(f.u, z)
    end
    return length(f)
end

"""
Creates a new block and returns its index.
"""
function newblock!(f::_VectorBlockFieldA{D, M, G, T}) where {D, M, G, T}
    S = M + 2G + 1
    append!(f.u, Iterators.repeated(zero(T), D * S^D))
    return length(f)
end

function newblock!(f::_VectorBlockFieldV{D, M, G, T}) where {D, M, G, T}
    S = M + 2G + 1
    MT = Tuple{ntuple(_ -> S, Val(D))..., D}
    z = zero(MArray{MT, Float64})
    push!(f.u, z)
    return length(f)
end

"""
Creates a series of new blocks returns the final length.
"""
function newblocks!(f::_VectorBlockFieldA{D, M, G, T}, n) where {D, M, G, T}
    S = M + 2G + 1
    append!(f.u, Iterators.repeated(zero(T), n * D * S^D))
    return length(f)
end

function newblocks!(f::_VectorBlockFieldV{D, M, G, T}, n) where {D, M, G, T}
    S = M + 2G + 1
    MT = Tuple{ntuple(_ -> S, Val(D))..., D}
    for i in 1:n
        z = zero(MArray{MT, Float64})
        push!(f.u, z)
    end
    return length(f)
end

"""
Create a new block in field `f` and checks that it is equal to `n`.
"""
function newblock!(f, n)
    n1 = newblock!(f)
    @assert n1 == n "Block number assertion failed.  Some fields are in an inconsistent state"
end


# Helper functions that should reduce to compile-time constants
Base.eltype(::AbstractBlockField{D, M, G, T}) where {D, M, G, T} = T

""" Return the size of each block (not counting ghost cells). """
blksize(::ScalarBlockField{D, M}) where {D, M} = ntuple(_ -> M, Val(D))
blksize(::VectorBlockField{D, M}) where {D, M} = tuple(ntuple(_ -> M + 1, Val(D))..., D)

""" Return the size of each block (counting ghost cells). """
blksizeghost(::ScalarBlockField{D, M, G}) where {D, M, G} = ntuple(_ -> M + 2G, Val(D))
blksizeghost(::VectorBlockField{D, M, G}) where {D, M, G} = tuple(ntuple(_ -> M + 2G + 1, Val(D))..., D)

""" Return number of valid (i.e. non-ghost) cells along each dimension. """
sidelength(::ScalarBlockField{D, M}) where {D, M} = M
sidelength(::VectorBlockField{D, M}) where {D, M} = M + 1

""" Return number of valid (i.e. non-ghost) cells along each dimension for scalar fields in the same
mesh. """
scalarlength(::ScalarBlockField{D, M}) where {D, M} = M
scalarlength(::VectorBlockField{D, M}) where {D, M} = M

""" Return number of ghost cells in perpendicular to each face. """
nghost(::ScalarBlockField{D, M, G}) where {D, M, G} = G
nghost(::VectorBlockField{D, M, G}) where {D, M, G} = G

""" Select the first half (`h==1`) or second half (`h==2`) of a range. """
function halfrange(r::UnitRange, h)
    if h == 1
        return first(r):(first(r) + div(length(r), 2) - 1)
    elseif h == 2
        return (first(r) + div(length(r), 2)):last(r)
    end
end


""" Return a CartesianIndices with the block indices ignoring ghosts."""
function localindices(::ScalarBlockField{D, M}) where {D, M}
    CartesianIndices(ntuple(d -> 1:M, Val(D)))
end

""" Return a CartesianIndices with the block indices of component `j` of a vector field 
    (only valid cells, not ghosts). """
function localindices(::VectorBlockField{D, M}, j) where {D, M}
    CartesianIndices((ntuple(d -> d == j ? (1:M + 1) : (1:M), Val(D))..., j:j))
end

""" Compute global cell indices from local ones `c` given block coordinates `b`. """
function localtoglobal(::ScalarBlockField{D, M}, c::CartesianIndex{D}, b::CartesianIndex{D}) where {D, M}
    c + M * (b - oneunit(b))
end

function localtoglobal(::ScalarBlockField{D, M}, c::CartesianIndices{D}, b::CartesianIndex{D}) where {D, M}
    c .+ M * (b - oneunit(b))
end

""" Shift indices to account for ghost indices. """
function addghost(f::ScalarBlockField{D, M, G}, c::CartesianIndex{D}) where {D, M, G}
    c + G * oneunit(c)
end

function addghost(f::ScalarBlockField{D, M, G}, c::CartesianIndices{D}) where {D, M, G}
    c .+ G * oneunit(c)
end

""" Ranges of indices excluding ghost cells. """
function validindices(f::ScalarBlockField{D, M, G}) where {D, M, G}
    CartesianIndices(ntuple(_ -> G + 1:G + M, Val(D)))
end

function validindices(f::VectorBlockField{D, M, G}, dim::Integer) where {D, M, G}
    CartesianIndices(ntuple(d -> d == dim ? ((G + 1):(G + M + 1)) : ((G + 1):(G + M)), Val(D)))
end

function subblockindices(f::ScalarBlockField{D, M, G}, sb::CartesianIndex) where {D, M, G}
    mhalf, r = divrem(M, 2)
    r == 0 || error("M must be divisible by 2")
    
    CartesianIndices(ntuple(d -> (G + 1 + (sb[d] - 1) * mhalf):(G + sb[d] * mhalf), Val(D)))
end


""" Indices of cells in the parent block for a refinement boundary. """
function subblockbnd(f::ScalarBlockField{D, M, G},
                     sb::CartesianIndex{D}, face::CartesianIndex{D}) where {D, M, G}
    sbinds = subblockindices(f, sb)
    ghalf, r = divrem(G, 2)
    r == 0 || error("G must be divisible by 2")
    
    function inrange(d)
        if face[d] == -1
            return sbinds.indices[d][1:ghalf]
        elseif face[d] == 1
            return sbinds.indices[d][(size(sbinds, d) - ghalf + 1):size(sbinds, d)]
        end

        @assert face[d] == 0 "Badly formed face indication"
        return sbinds.indices[d][axes(sbinds, d)]
    end
    
    CartesianIndices(ntuple(d->inrange(d), Val(D)))
end

# Handling of block boundaries

"""
Check that the index is a correct face specification. 

A CartesianIndex is a correct face specification if it consists only on
(-1, 0, 1) but not all indices are 0.
"""
@generated function isface(face::CartesianIndex{D}) where {D}
    quote
        if @nany $D d->(face[d] < -1 || face[d] > 1)
            return false
        end
        return @nany $D d->(face[d] != 0)
    end
end


"""
Perpendicular direction to a face.  Assumes that there is only one element in face that is non-zero
and returns its index.
"""
@generated function perpdim(face::CartesianIndex{D}) where {D}
    quote
        @nif $D d->(face[d] != 0) (d->return d)
    end
end

"""
Check that a face is a direction, i.e. that one and only one component is nonzero.
"""
isaxis(face::CartesianIndex) = 1 == count(!=(0), Tuple(face))

# First and last indices of the ghost area
ghostfirst(::ScalarBlockField{D, M, G}, k) where {D, M, G} = k == -1 ? 1 : (M + G + 1)
ghostlast(::ScalarBlockField{D, M, G}, k) where {D, M, G} = k == -1 ? G : M + 2G
ghostrange(f::ScalarBlockField, k) = ghostfirst(f, k):ghostlast(f, k)

# Same but for non-ghost (valid) cells
validfirst(::ScalarBlockField{D, M, G}) where {D, M, G} = G + 1
validlast(::ScalarBlockField{D, M, G}) where {D, M, G} = G + M
validrange(f::ScalarBlockField) = validfirst(f):validlast(f)
validrange2(f::ScalarBlockField) = validfirst(f):2:validlast(f)

# For region that overlap with ghost neighbors
overlapfirst(::ScalarBlockField{D, M, G}, k) where {D, M, G} = k == -1 ? (G + 1) : (M + 1)
overlaplast(::ScalarBlockField{D, M, G}, k) where {D, M, G} = k == -1 ? 2G : M + G
overlaprange(f::ScalarBlockField, k) = overlapfirst(f, k):overlaplast(f, k)

# Index of the boundary face for vector fields
bndindex(::VectorBlockField{D, M, G}, k) where {D, M, G} = k == -1 ? (G + 1) : (G + M + 1)

""" Return a CartesianIndices for ghost cells at a given face provided as a CartesianIndex. """
@inline function ghostindices(f::ScalarBlockField{D}, face::CartesianIndex{D}) where {D}
    @assert isface(face)
    
    CartesianIndices(ntuple(dim -> (face[dim] == 0 ? validrange(f) :
                                    ghostrange(f, face[dim])), Val(D)))
end


""" 
Return a CartesianIndices for the cells that overlap with ghost from a neighbor at
face `face`. 
"""
@inline function overlapindices(f::ScalarBlockField{D}, face::CartesianIndex{D}) where {D}
    @assert isface(face)
    
    CartesianIndices(ntuple(dim -> (face[dim] == 0 ? validrange(f) :
                                    overlaprange(f, face[dim])), Val(D)))
end


"""
Computes the "mirror" index of `i` mapping non-ghost to ghost (and viceversa).
`dir` must be -1 (lowest boundary) or +1 (highest boundary).
"""
function mirrorghost(f::ScalarBlockField{D, M, G}, i::Int, dir) where {D, M, G}
    if dir == -1
        return 2G - i + 1
    elseif dir == 1
        return 2 * (G + M) - i + 1
    end

    return i
end


"""
Computes the "mirror" index of a range possibly mapping non-ghost to ghost 
(and viceversa). `dir` must be -1 (lowest boundary) or +1 (highest boundary)
or 0 (no change).
"""
function mirrorghost(f::ScalarBlockField, r::AbstractRange, dir) 
    r, rev = promote(r, reverse(mirrorghost(f, last(r), dir):mirrorghost(f, first(r), dir)))
    return (dir == 0 ? r : rev)
end


"""
Computes the "mirror" indices of CartesianIndices.
"""
function mirrorghost(f::ScalarBlockField{D}, ci::CartesianIndices{D},
                     dir::CartesianIndex{D}) where {D}
    CartesianIndices(ntuple(d->mirrorghost(f, ci.indices[d], dir[d]), Val(D)))
end

"""
Compute an `CartesianIndices` for the boundary face of a VectorBlockField
"""
function bndface(f::VectorBlockField{D, M, G}, face::CartesianIndex{D}) where {D, M, G}
    @assert isface(face)
    CartesianIndices(ntuple(dim -> (face[dim] == 0 ?
                                    ((G + 1):(G + M)) :
                                    (bndindex(f, face[dim]):bndindex(f, face[dim]))), Val(D)))
end


"""
Copies all blocks in the `layer` from `src` to `dest`
"""
function copyto!(dest, src, layer::BlockLayer)
    @batch for (_, blk) in layer.pairs
        b1 = dest[blk]
        b2 = src[blk]
        @turbo b1 .= b2
    end
end


"""
Writes into `dest` the difference `a1` - `a2`.
"""
function diffto!(dest, a1, a2, layer::BlockLayer)
    @batch for (_, blk) in layer.pairs
        destblk = dest[blk]
        a1blk = a1[blk]
        a2blk = a2[blk]
        @turbo destblk .= a1blk .- a2blk
    end
end



"""
Traverses a tree of Broadcasted elements and changes each Broadcast type with `T` over element `i`
of the underlying ScalarBlockField. 
"""
@inline function mapbroadcast(T::Type{Z}, i, x::Broadcast.Broadcasted) where {Z}
    Broadcast.Broadcasted{T}(x.f, mapbroadcast(T, i, x.args))
end

@inline mapbroadcast(T::Type{Z}, i, args::Tuple{}) where {Z} = ()
@inline function mapbroadcast(T::Type{Z}, i, args::Tuple) where {Z}
    (mapbroadcast(T, i, args[1]), mapbroadcast(T, i, Base.tail(args))...)
end
@inline mapbroadcast(T, i, x::AbstractBlockField) = x[i]
#mapbroadcast(T, i, x::StrippedBlockField) = x.bf[i]
@inline mapbroadcast(T, i, x::Any) = x



