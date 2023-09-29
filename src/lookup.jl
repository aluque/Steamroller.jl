# Code for lookup tables with arbitrary transformations in X and Y.

struct LookupTable{TX, TY, F, G, GI}
    # X data (f(x)).  Much faster when this is a uniform range
    fx::TX

    # Y daya (g(y)).
    gy::TY

    # The transform function for x
    f::F

    # The transform function for y
    g::G

    # The inverse of g
    ginv::GI

    function LookupTable(fx, gy; f=identity, g=identity, ginv=identity)
        if gy isa Vector{<:Real}
            @assert length(fx) == length(gy)
        else
            @assert all(==(length(fx)), length.(gy))
        end
        
        new{typeof(fx), typeof(gy), typeof(f), typeof(g), typeof(ginv)}(fx, gy, f, g, ginv)
    end        
end


@inline Base.@assume_effects :foldable function prefetch(tbl::LookupTable, x)
    fx = tbl.f(x)
    i = searchsortedlast(tbl.fx, fx)
    
    if i == 0
        # Allow constant-value extrapolation below the range
        return (1, zero(promote_type(eltype(tbl.fx), typeof(fx))))
    end
    
    @boundscheck checkbounds(tbl.fx, i)
    @boundscheck checkbounds(tbl.fx, i + 1)
    
    @inbounds w = (fx - tbl.fx[i]) / (tbl.fx[i + 1] - tbl.fx[i])
    return (i, w)
end

_prefetch(tbl, x, ::Nothing) = prefetch(tbl, x)
_prefetch(tbl, x, p) = p

@inline function (tbl::LookupTable)(x; prefetch=nothing)
    (i, w) = _prefetch(tbl, x, prefetch)

    @inbounds gy = w * tbl.gy[i + 1] + (1 - w) * tbl.gy[i]

    return tbl.ginv(gy)
end

@inline function (tbl::LookupTable)(x, col::Integer; prefetch=nothing)
    (i, w) = _prefetch(tbl, x, prefetch)

    @inbounds gy = w * tbl.gy[col][i + 1] + (1 - w) * tbl.gy[col][i]

    return tbl.ginv(gy)
end


"""    
    Guess a range from a vector `v`.  Raises a warning if the values are too far 
    from uniform.
"""
function approxrange(v::AbstractVector; atol::Real=0, rtol::Real=0.001)
    @assert issorted(v)
    
    h = diff(v)
    hmean = sum(h) / length(h)
    
    if !all(x -> isapprox(x, hmean; atol, rtol), h)
        # Note type instability
        return v
    end

    l = length(v)
    
    return LinRange(v[begin], v[end], l)
end


"""
    Load a lookuptable from a delimited file.
"""
function loadtable(fname::AbstractString, T::Type=Float64; resample_into=nothing,
                   xcol=1, ycol=2, f=identity, g=identity, ginv=identity, kw...)
    data = readdlm(fname, T)
    
    fx = approxrange(f.(data[:, xcol]); kw...)
    gy = g.(data[:, ycol])

    if !isnothing(resample_into)
        fx1 = LinRange(fx[begin], fx[end], resample_into)
        gy1 = linear_interpolation(fx, gy).(fx1)

        (fx, gy) = (fx1, gy1)
    end
    
    LookupTable(fx, gy; f, g, ginv)
end


"""
    Load a multi-column lookuptable from a series of columns.
    If you have a table as a DataFrame pass `tbl=eachcol(df)` in order to support regular indexing
    by-columns.
"""
function loadtable(tbl, T::Type=Float64; resample_into=nothing,
                   xcol=1, ycols=nothing, f=identity, g=identity, ginv=identity, kw...)
    fx = approxrange(f.(tbl[xcol]); kw...)
    ycols = isnothing(ycols) ? keys(tbl) : ycols
    
    gy = [g.(tbl[ycol]) for ycol in ycols]
    
    if !isnothing(resample_into)
        fx1 = LinRange(fx[begin], fx[end], resample_into)
        gy1 = [linear_interpolation(fx, igy).(fx1) for igy in gy]

        (fx, gy) = (fx1, gy1)
    end
    
    LookupTable(fx, gy; f, g, ginv)
end

