#= Stencils 

Stencils are defined implementing the iterator interface and they are coded
for speed.
=#

""" "Star"-type stencil. Does not include diagonals. """
struct StarStencil{D}; end

""" "Box"-type stencil. Includes diagonals. """
struct BoxStencil{D}; end

""" "Cube"-type stencil: 2^D elements with values 1 or 2. """
struct CubeStencil{D, K}; end

function Base.iterate(::StarStencil{D}) where D
    Base.setindex(zero(CartesianIndex{D}), -1, 1), (-1, 1)
end

function Base.iterate(::StarStencil{D}, (val, dim)) where D
    val += 2

    if val > 1
        val = -1
        dim += 1
    end

    if dim > D
        return nothing
    end

    Base.setindex(zero(CartesianIndex{D}), val, dim), (val, dim)
end


function Base.iterate(::BoxStencil{D}) where D
    iterfirst = CartesianIndex(ntuple(d -> -1, Val(D)))
    iterfirst, iterfirst
end


function Base.iterate(::BoxStencil{D}, state) where D
    valid, I = __incbox(state.I)
    
    valid || return nothing

    # drop 0,0,...
    # having this check reduces performance quite a bit but I didn't find
    # a way to prevent this.  Removing it from here and having a check
    # downstream is not better (usually a bit worse).
    if __allzeros(I)
        valid, I = __incbox(I)
    end

    return CartesianIndex(I...), CartesianIndex(I...)
end

function Base.iterate(::CubeStencil{D, K}) where {D, K}
    iterfirst = CartesianIndex(ntuple(d -> K, Val(D)))
    iterfirst, iterfirst
end


function Base.iterate(::CubeStencil{D, K}, state) where {D, K}
    valid, I = __inccube(state.I, K)
    valid || return nothing    
    return CartesianIndex(I...), CartesianIndex(I...)
end


# This recursive increse of indices is based upon CartesianIndices in the julia
# Base (see multidimensional.jl in Base).
__incbox(::Tuple{}, ::Tuple{}) = false, ()
@inline function __incbox(state::Tuple{Int})
    I = state[1] + 1

    valid = state[1] != 1
    return valid, (I, )
end


@inline function __incbox(state::Tuple{Int, Int, Vararg{Int}})
    I = state[1] + 1
        
    if state[1] != 1
        return true, (I, Base.tail(state)...)
    end
    
    valid, I = __incbox(Base.tail(state))
    return valid, (-1, I...)
end


@inline function __allzeros(t::Tuple{Int})
    return t[1] == 0
end

@inline function __allzeros(t::Tuple{Int, Int, Vararg{Int}})
    return t[1] == 0 && __allzeros(Base.tail(t))
end

__inccube(::Tuple{}, ::Tuple{}) = false, ()
@inline function __inccube(state::Tuple{Int}, K)
    I = state[1] + 1

    valid = state[1] != K + 1
    return valid, (I, )
end


@inline function __inccube(state::Tuple{Int, Int, Vararg{Int}}, K)
    I = state[1] + 1
        
    if state[1] != K + 1
        return true, (I, Base.tail(state)...)
    end
    
    valid, I = __inccube(Base.tail(state), K)
    return valid, (K, I...)
end


Base.eltype(::Type{StarStencil{D}}) where D = CartesianIndex{D}
Base.eltype(::Type{BoxStencil{D}}) where D = CartesianIndex{D}
Base.eltype(::Type{CubeStencil{D}}) where D = CartesianIndex{D}
Base.length(::StarStencil{D}) where D = 2D
Base.length(::BoxStencil{D}) where D = 3^D - 1
Base.length(::CubeStencil{D}) where D = 2^D
