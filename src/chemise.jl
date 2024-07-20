#=
This is a code to handle complex chemical networks with rate coefficients that depend on
external variables such as electric field.

The code aims to be performant and benefit from the julia compiler to dynamically produce code
that is tailored to the network topology.  To achieve that we encode this topology in the types
of the relevant data structs.  Then the `derivatives` function is @generated with this info on hand.

Define chemical models with the syntax

rs = ReactionSet(react"A + B -> C + D" => rate1,
                 react"A + C -> B + D" => rate2,...)

rate1, rate2 can be `Real`s, functions such as eabs -> c * exp(-e1/eabs), or anything that has a method
evalk(rate, args...).
=#

"""
A struct to contain the reactants and products, with their multiplicities.

To have performant schemes, we encode the reactants/products and their stochiometric coefficient
in the type, in tuples of 2-element tuples.
"""
struct Signature{L, R}
    function Signature(L::Tuple{Vararg{Tuple{Symbol, Int}}},
                       R::Tuple{Vararg{Tuple{Symbol, Int}}})
        return new{L, R}()
    end    
end

lhs(::Signature{L, R}) where {L, R} = L
lhs(::Type{Signature{L, R}}) where {L, R} = L
rhs(::Signature{L, R}) where {L, R} = R
rhs(::Type{Signature{L, R}}) where {L, R} = R


"""
A reaction with a rate.

The type G may contain a symbol that identifies different 'groups' of reactions. Sometimes you may want
to activate only certain groups, for example when computing derivatives pre/post photo-ioniation.
"""
struct Reaction{S <: Signature, K, G}
    signature::S
    k::K
    function Reaction(signature::S, k::K, G::Union{Symbol, Bool, Nothing}) where {S <: Signature, K}
        new{S, K, G}(signature, k)
    end
    function Reaction(signature::S, k::K) where {S <: Signature, K}
        new{S, K, nothing}(signature, k)
    end
end

Reaction(s::String, k, g=nothing) = Reaction(parse(Signature, s), k, g)

isactive(::Type{Reaction{S, H, G}}, groups::Union{Tuple, Int}) where {S, H, G} = isempty(groups) || G in groups
isactive(::Type{Reaction{S, H, G}}, groups::Symbol) where {S, H, G} = groups == G

signature(::Reaction{S}) where S = S
signature(::Type{Reaction{S, K, G}}) where {S, K, G} = S
lhs(::Reaction{S}) where {S}= lhs(S)
lhs(::Type{Reaction{S, K, G}}) where {S, K, G} = lhs(S)
rhs(::Reaction{S}) where {S} = rhs(S)
rhs(::Type{Reaction{S, K, G}}) where {S, K, G} = rhs(S)


evalk(r::Reaction, args...; prefetch=nothing) = _evalk(prefetch, r.k, args...)
_evalk(::Nothing, r, args...) = evalk(r, args...)
_evalk(prefetch, r, args...) = evalk(r, args...; prefetch)
       
evalk(f::Real, args...) = f
evalk(f::Function, args...) = f(args...)

varname(::Type{Reaction{S, K, G}}) where {S, K, G} = varname(K)

varname(::Type{T}) where T <: Real = nothing
varname(::Type{T}) where T <: Function = nothing
varname(::Type{T}) where T = nothing


"""
Wrap a rate and a bibliography reference.
"""
struct Biblio{K, S <: AbstractString}
    k::K
    ref::S
end

evalk(f::Biblio, args...) = evalk(f.k, args...)


"""
A rate taken from a lookup table.

The parameter `H` is a hash of the lookup table; this allows to for compile-time prefetch of the lookup.
"""
struct RateLookup{L, H}
    lookup::L
    index::Int

    function RateLookup(lookup::L, index::Int) where L
        H = hash(L)
        new{L, H}(lookup, index)
    end
end

function RateLookup(lookup, fname::String)
    index = addcol(lookup, fname)
    return RateLookup(lookup, index)
end

function RateLookup(lookup, index::Symbol)
    i = findfirst(==(index), lookup.colnames)
    isnothing(i) && throw(ArgumentError("Column $(index) not found in the lookup table"))

    RateLookup(lookup, i)
end


@inline evalk(f::RateLookup, args...; prefetch=nothing) = @inline f.lookup(args..., f.index; prefetch)
@inline prefetch(f::RateLookup, args...) = prefetch(f.lookup, args...)
@inline prefetch(r::Reaction, args...) = prefetch(r.k, args...)

varname(::Type{RateLookup{L, H}}) where {L, H} = string(H)


"""
A struct to contain a reaction set.

The species are encoded in the type `S` whereas reactions are a tuple encoded in `R`. `F` contains
a possible set of 'fixed' densities with values in the `fixed` field.
"""
struct ReactionSet{S, R, F, FT}
    reactions::R
    fixedval::FT
    
    function ReactionSet(species::Tuple{Vararg{Symbol}},
                         reactions::Tuple{Vararg{<:Reaction}},
                         fixed::Tuple{Vararg{Symbol}},
                         fixedval)
        new{species, typeof(reactions), fixed, typeof(fixedval)}(reactions, fixedval)
    end
end

function ReactionSet(reactions::Pair...; fixed=(), fixedval=(), fix=nothing)
    if !isnothing(fix)
        fixed = (fixed..., first.(fix)...)
        fixedval = (fixedval..., last.(fix)...)
    end

    _reactions = vcat(map(((g, r),) -> map(((sig, k),) -> Reaction(sig, k, g), r), reactions)...)
    species = tuple(list_of_species(_reactions, fixed)...)
    
    return ReactionSet(species, tuple(_reactions...), fixed, fixedval)
end

ReactionSet(reactions::Vector; kw...) = ReactionSet(nothing => reactions; kw...)


nspecies(rs::ReactionSet{S}) where {S} = length(S)

@generated function species_charge(::ReactionSet{S}) where S
    guess_charge(y) = y == :e ? -1 : count(==('+'), string(y)) - count(==('-'), string(y))
    expr = :(SA[])
    for spec in S
        push!(expr.args, guess_charge(spec))
    end
    return expr
end

function list_of_species(reactions, fixed)
    a = Set{Symbol}()
    for r in reactions
        for (s, _) in lhs(r)
            if isnothing(speciesindex(fixed, s))
                push!(a, s)
            end
        end

        for (s, _) in rhs(r)
            if isnothing(speciesindex(fixed, s))
                push!(a, s)
            end
        end
    end

    # Electrons are always first
    if :e in a
        delete!(a, :e)
        return [:e; sort!(collect(a))]
    else
        return sort!(collect(a))
    end
end


species(rs::ReactionSet{S}) where S = S
idx(rs::ReactionSet{S}, s::Symbol) where S = speciesindex(S, s)
idx(rs::ReactionSet{S}, s::String) where S = speciesindex(S, Symbol(s))

fixed_species(rs::ReactionSet{S, R, F}) where {S, R, F} = F
speciesindex(S, s::Symbol) = findfirst(==(s), S)

evalfixed(v::Real, x) = v
evalfixed(f::Function, x) = f(x)

function _rate_expr(S, F, L, idx, pref)
    if isnothing(pref)
        expr = :(*(evalk(rs.reactions[$idx], args...)))
    else
        v = Symbol("prefetch_" * pref)
        expr = :(*(evalk(rs.reactions[$idx], args...; prefetch=$v)))
    end
    
    for (spec, coeff) in L
        i = speciesindex(S, spec)
        if !isnothing(i)            
            push!(expr.args, coeff == 1 ? :(n[$i]) : :(n[$i]^$coeff))
            continue
        end
        
        i = speciesindex(F, spec)
        if !isnothing(i)
            push!(expr.args, coeff == 1 ?
                :(evalfixed(rs.fixedval[$i], x)) :
                :(evalfixed(rs.fixedval[$i], x)^$coeff))
        end
    end
    return expr
end

function _derivs_expr(S, R)
    expr = :(SA[])
    for (i, spec) in enumerate(S)
        expri = :(+())
        for (j, r) in enumerate(R.parameters)
            c = 0
            for (spec1, coeff) in lhs(r)
                spec1 == spec || continue
                c -= coeff
            end
            for (spec1, coeff) in rhs(r)
                spec1 == spec || continue
                c += coeff
            end
            if c != 0
                push!(expri.args, :($c * _rates[$j]))
            end            
        end
        push!(expr.args, length(expri.args) > 1 ? expri : 0)
    end
    return expr
end

function _prefetching(R, G)
    expr = quote end
    prefetched = Set{String}()
    for (i, r) in enumerate(R.parameters)
        isactive(r, G) || continue
        
        v = varname(r)
        if !isnothing(varname(r)) && !(v in prefetched)
            push!(prefetched, v)
            v = Symbol("prefetch_" * v)
            push!(expr.args, :($v = prefetch(rs.reactions[$i], args...)))
        end        
    end
    return expr
end

"""
Compute derivatives for the reaction set `rs` with given densities `n` and external variables
`args...` (e.g. electric field). `groups` is a possible set of reaction groups that we want to
(exclusively) activate; it must be provided as `Val((:group1, :group2...))` or Val(group).
`x` is a location; it can be ignored or used to set inhomogeneous fixed values.
"""
@generated function derivs(rs::ReactionSet{S, R, F}, groups::Val{G}, n::AbstractVector, x,
                            args...) where {S, R, F, G}
    rates = :(())
    for (idx, r) in enumerate(R.parameters)
        if isactive(r, G)
            pref = varname(r)
            push!(rates.args, _rate_expr(S, F, lhs(r), idx, pref))
        else
            push!(rates.args, 0)
        end
    end
    pref = _prefetching(R, G)
    dexpr = _derivs_expr(S, R)
    expr = quote
        $pref
        _rates = $rates
        _dn = $dexpr
        return _dn
    end
    return expr
end

derivs(rs::ReactionSet{S, R, F}, n::AbstractVector, x, args...) where {S, R, F} = derivs(rs, Val{()}(), n, x, args...)


function Base.parse(::Type{Signature}, signature)
    arrow = r"\s*[-=]+>\s*"
    plus = r"\s+\+\s+"

    (lhs, rhs) = (filter(s -> length(s) > 0, split(x, plus)) for x in split(signature, arrow))
    lhs, rhs = [[_multiplicity(item) for item in x] for x in (lhs, rhs)]

    return Signature(Tuple(lhs), Tuple(rhs))
end

function Base.show(io::IO, r::Signature{L, R}) where {L, R}
    maybemult(m, s) = m > 1 ? "$m * $s" : "$s"
    
    lhs_str = join(map(((s, m),) -> maybemult(m, s), L), " + ")
    rhs_str = join(map(((s, m),) -> maybemult(m, s), R), " + ")
    print(io, "react\"", lhs_str, " -> ", rhs_str, "\"")
end

function Base.show(io::IO, rs::ReactionSet{S, R}) where {S, R}
    for r in rs.reactions
        println(io, r)
    end
end

function Base.show(io::IO, rs::Reaction)
    print(io, rs.signature)
    print(io, " [k = ")
    print(io, rs.k)
    print(io, "]")
end

function _multiplicity(code)
    rx = r"((\d+)\s*\*\s*)?([\w\(\).^+*-]+)"
    m = match(rx, code)
    (n1, s) = (m.captures[2], m.captures[3])
    
    n = isnothing(n1) ? 1 : parse(Int, n1)

    return (Symbol(s), n)
end

macro react_str(str)
    parse(Signature, str)
end
