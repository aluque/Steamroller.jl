#=
   Routines to handle fields stored in block decks.
=#

greetme() = "Hello World hello now"

"""
A block field stored in a series of MArrays
Encoded in a type for performance reasons.
`D`: Number of dimensions
`M`: Side length in grid points
`G`: Number of ghost cells
`A`: MArray type
"""
struct ScalarBlockField{D, M, G, T, N, A} <: AbstractVectorOfArray{T, N, A}
    u::A

    function ScalarBlockField{D, M, G, T}() where {D, M, G, T}
        @assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G
        N = D + 1
        val = MArray{NTuple{D, S}, T, D, S^D}[]
        new{D, M, G, T, N, typeof(val)}(val)
    end

    function ScalarBlockField{D, M, G, T}(len) where {D, M, G, T}
        @assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G
        N = D + 1
        val = [zero(MArray{NTuple{D, S}, Float64}) for i in 1:len]
        new{D, M, G, T, N, typeof(val)}(val)
    end
end

"""
A vector block field encoded in a series of MArrays.

Each MArray corresponding to a block has dimensions
(M + 2G + 1, M + 2G + 1, ..., D).
Usually we use this to store a vector field evaluated at cell
interfaces and related to a gradient of a scalar field.    
"""
struct VectorBlockField{D, M, G, T, N, A} <: AbstractVectorOfArray{T, N, A}
    # We cannot construct the type here so it must be parametric
    u::A

    function VectorBlockField{D, M, G, T}() where {D, M, G, T}
        @assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G + 1
        MT = Tuple{ntuple(_ -> S, Val(D))..., D}
        N = D + 2
        val = MArray{MT, T}[]
        new{D, M, G, T, N, typeof(val)}(val)
    end

    function VectorBlockField{D, M, G, T}(len) where {D, M, G, T}
        @assert rem(M, 2) == 0 "Block size must be divisible by 2"

        S = M + 2G + 1
        MT = Tuple{ntuple(_ -> S, Val(D))..., D}
        N = D + 2
        val = zeros(MArray{MT, T}, len)
        new{D, M, G, T, N, typeof(val)}(val)
    end
end


# Most frequently we have to dispatch with D,  D and M or D, M, G so we define shortcuts
const BlockField = Union{ScalarBlockField, VectorBlockField}

getblk(f::BlockField, blk) = f.u[blk]
valid(f::ScalarBlockField, blk) = view(f.u[blk], validindices(f))

"""
Creates a new block and returns its index.
"""
function newblock!(f::ScalarBlockField{D, M, G}) where {D, M, G}
    S = M + 2G
    z = zero(MArray{NTuple{D, S}, Float64})
    push!(f.u, z)
    return length(f.u)
end

"""
Creates a series of new blocks returns the final length.
"""
function newblocks!(f::ScalarBlockField{D, M, G}, n) where {D, M, G}
    S = M + 2G
    for i in 1:n
        z = zero(MArray{NTuple{D, S}, Float64})
        push!(f.u, z)
    end
    return length(f.u)
end

"""
Creates a new block and returns its index.
"""
function newblock!(f::VectorBlockField{D, M, G}) where {D, M, G}
    S = M + 2G + 1
    MT = Tuple{ntuple(_ -> S, Val(D))..., D}
    
    z = zero(MArray{MT, Float64})
    push!(f.u, z)
    return length(f.u)
end

"""
Creates a series of new blocks returns the final length.
"""
function newblocks!(f::VectorBlockField{D, M, G}, n) where {D, M, G}
    S = M + 2G + 1
    MT = Tuple{ntuple(_ -> S, Val(D))..., D}
    for i in 1:n
        z = zero(MArray{MT, Float64})
        push!(f.u, z)
    end
    return length(f.u)
end



"""
Delete block at index `blk` by moving the last block into `blk`
"""
function Base.deleteat!(f::BlockField, blk)
    last = pop!(f.u)

    # Did we remove the last block?
    (blk > length(f.u)) && return

    f.u[blk] = last
end

# Helper functions that should reduce to compile-time constants
Base.eltype(::ScalarBlockField{D, M, G, T, A}) where {D, M, G, T, A} = T
Base.eltype(::VectorBlockField{D, M, G, T, A}) where {D, M, G, T, A} = T

""" Return the size of each block (not counting ghost cells). """
blksize(::ScalarBlockField{D, M}) where {D, M} = ntuple(_ -> M, Val(D))
blksize(::VectorBlockField{D, M}) where {D, M} = tuple(ntuple(_ -> M + 1, Val(D))..., D)

""" Return the size of each block (counting ghost cells). """
blksizeghost(::ScalarBlockField{D, M, G}) where {D, M, G} = ntuple(_ -> M + 2G, Val(D))
blksizeghost(::VectorBlockField{D, M, G}) where {D, M, G} = tuple(ntuple(_ -> M + 2G + 1, Val(D))..., D)

""" Return number of valid (i.e. non-ghost) cells along each dimension. """
sidelength(::ScalarBlockField{D, M}) where {D, M} = M
sidelength(::VectorBlockField{D, M}) where {D, M} = M + 1

""" Return number of ghost cells in perpendicular to each face. """
nghost(::ScalarBlockField{D, M, G}) where {D, M, G} = G
nghost(::VectorBlockField{D, M, G}) where {D, M, G} = G

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


""" Return a CartesianIndices for ghost cells at a given face provided as a CartesianIndex. """
function ghostindices(f::ScalarBlockField{D}, face::CartesianIndex{D}) where {D}
    @assert isface(face)
    
    CartesianIndices(ntuple(dim -> (face[dim] == 0 ? validrange(f) :
                                    ghostrange(f, face[dim])), Val(D)))
end


""" 
Return a CartesianIndices for the cells that overlap with ghost from a neighbor at
face `face`. 
"""
function overlapindices(f::ScalarBlockField{D}, face::CartesianIndex{D}) where {D}
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
