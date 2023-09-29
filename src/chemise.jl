#=
This is a code to handle complex chemicat networks with rate coefficients that depend on
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
"""
struct Reaction{S <: Signature, K}
    signature::S
    k::K
    function Reaction(signature::S, k::K) where {S <: Signature, K}
        new{S, K}(signature, k)
    end
end

Reaction(s::String, k) = Reaction(parse(Signature, s), k)
signature(::Reaction{S}) where S = S
signature(::Type{Reaction{S, K}}) where {S, K} = S
lhs(::Reaction{S}) where {S}= lhs(S)
lhs(::Type{Reaction{S, K}}) where {S, K} = lhs(S)
rhs(::Reaction{S}) where {S} = rhs(S)
rhs(::Type{Reaction{S, K}}) where {S, K} = rhs(S)


evalk(r::Reaction, args...) = evalk(r.k, args...)
evalk(f::Real, args...) = f
evalk(f::Function, args...) = f(args...)


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
"""
struct RateLookup{L}
    lookup::L
    index::Int
end

evalk(f::RateLookup, args...) = f.lookup(args..., f.index, prefetch=prefetch(f.lookup, args...))


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

function ReactionSet(reactions::Pair{<:Signature, <:Any}...; fixed=(), fixedval=SA[])
    species = tuple(list_of_species(map(first, reactions), fixed)...)
    _reactions = map(((s, k),) -> Reaction(s, k), reactions)
    return ReactionSet(species, tuple(_reactions...), fixed, fixedval)
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
fixed_species(rs::ReactionSet{S, R, F}) where {S, R, F} = F
speciesindex(S, s::Symbol) = findfirst(==(s), S)

function rate_expr(S, F, L, idx)
    expr = :(*(evalk(rs.reactions[$idx], args...)))
    for (spec, coeff) in L
        i = speciesindex(S, spec)
        if !isnothing(i)
            push!(expr.args, coeff == 1 ? :(n[$i]) : :(n[$i]^$coeff))
            continue
        end
        
        i = speciesindex(F, spec)
        if !isnothing(i)
            push!(expr.args, coeff == 1 ? :(rs.fixedval[$i]) : :(rs.fixedval[$i]^$coeff))
        end        
    end
    return expr
end

function derivs_expr(S, R)
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

@generated function derivatives(rs::ReactionSet{S, R, F}, n, args...) where {S, R, F}
    rates = :(())
    for (idx, r) in enumerate(R.parameters)
        push!(rates.args, rate_expr(S, F, lhs(r), idx))
    end

    dexpr = derivs_expr(S, R)
    expr = quote
        _rates = $rates
        _dn = $dexpr
        return _dn
    end
    return expr
end


function Base.parse(::Type{Signature}, signature)
    arrow = r"\s*[-=]+>\s*"
    plus = r"\s+\+\s+"

    (lhs, rhs) = (split(x, plus) for x in split(signature, arrow))
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
