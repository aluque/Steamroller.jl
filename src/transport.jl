#All around transport models (including impact ionization and attachment).

abstract type AbstractTransportModel end

"""
A minimal transport model with constant mobility and diffusion coefficient. Ionization follows
the townsend expression.
"""
@kwdef struct MinimalTransportModel{T} <: AbstractTransportModel
    "Mobility."
    μ::T

    "Diffusion coefficient."
    D::T

    "α for ionization."
    αi::T

    "Characteristic field for ionization."
    Ei::T

    "α for attachment."
    αa::T

    "Characteristic field for attachment."
    Ea::T
end


@inline mobility(m::MinimalTransportModel, eabs, p=nothing) = m.μ
@inline diffusion(m::MinimalTransportModel, eabs, p=nothing) = m.D
@inline nettownsend(m::MinimalTransportModel, eabs, p=nothing) = townsend(m, eabs) - attachment(m, eabs)
@inline attachment(m::MinimalTransportModel, eabs, p=nothing) = m.αa * exp(-m.Ea / eabs)
@inline townsend(m::MinimalTransportModel, eabs, p=nothing) = m.αi * exp(-m.Ei / eabs)
@inline netionization(m::MinimalTransportModel, eabs, p=nothing) = m.μ * eabs * nettownsend(m, eabs)


"""
The transport model proposed in Bagheri et al. 2018.
"""
@kwdef struct BagheriTransportModel{T} <: AbstractTransportModel
    "μ0 in μe = μ0 |E|^eμ."
    μ0::T = 2.3987

    "eμ in μe = μ0 |E|^eμ."
    eμ::T = -0.26

    "D0 in De = D0 |E|^eD."
    D0::T = 4.3628e-3

    "D0 in De = D0 |E|^eD."
    eD::T = 0.22

    "α0 in α = (α0 + α1 / |E|^eα) * exp(mα / |E|)"
    α0::T = 1.1944e6
    
    "α1 in α = (α0 + α1 / |E|^eα) * exp(mα / |E|)"
    α1::T = 4.3666e26

    "eα in α = (α0 + α1 / |E|^eα) * exp(mα / |E|)"
    eα::Int = 3

    "mα in α = (α0 + α1 / |E|^eα) * exp(mα / |E|)"
    mα::T = -2.73e7

    "Attachment rate"
    η::T = 340.75

    # Seems like the CWI group actually used
    # η::T = 340.98719
end

"""
When making multiple queries to the transport data, we may want to pre-fetch some data for performance
reasons.  Some models simply ignore this.
"""
@inline prefetch(::BagheriTransportModel, eabs) = nothing

"""
Velocity in transport model `m` for an electric field `eabs`.
"""
@inline mobility(m::BagheriTransportModel, eabs, p=nothing) = @fastmath m.μ0 * eabs^m.eμ
#@inline mobility(m::BagheriTransportModel, eabs) = 0.0372193


"""
Diffusion rate in transport model `m` for an electric field `eabs`.
"""
@inline diffusion(m::BagheriTransportModel, eabs, p=nothing) = @fastmath m.D0 * eabs^m.eD


"""
Townsend coefficient (excluding attachment) in transport model `m` for an electric field `eabs`.
"""
@inline function townsend(m::BagheriTransportModel, eabs, p=nothing)
    (;α0, α1, eα, mα, η) = m

    @fastmath (α0 + α1 / eabs^eα) * exp(mα / eabs)
end

"""
Townsend attachment coefficient in transport model `m` for an electric field `eabs`.
"""
@inline attachment(m::BagheriTransportModel, eabs, p=nothing) = m.η

"""
Townsend coefficient in transport model `m` for an electric field `eabs`.
"""
@inline function nettownsend(m::BagheriTransportModel, eabs, p=nothing)
    @fastmath townsend(m, eabs) - attachment(m, eabs)
end


"""
Net ionization in transport model `m` for an electric field `eabs`.
"""
@inline function netionization(m::BagheriTransportModel, eabs, p=nothing)
    (;α0, α1, eα, mα, η) = m

    α = nettownsend(m, eabs)
    α * eabs * mobility(m, eabs)
end


########################################
# Interpolation from CWI in the supp. material of Bagheri et al. 2018
########################################
struct CWITransportModel{T, T1, T2, T3} <: AbstractTransportModel
    # These are lookuptables
    eta::T
    alpha::T1
    mu::T2
    dif::T3
    
    function CWITransportModel{T}(alpha_fname, mu_fname, dif_fname; resample_into=1000) where T
        # alpha_tbl = loadtable(alpha_fname, f=Base.FastMath.log10)
        # mu_tbl = loadtable(mu_fname, f=Base.FastMath.log10)
        # dif_tbl = loadtable(dif_fname, f=Base.FastMath.log10)

        alpha_tbl = loadtable(alpha_fname; resample_into)
        mu_tbl = loadtable(mu_fname; resample_into)
        dif_tbl = loadtable(dif_fname; resample_into)
        
        # This correction factor is to prevent roundoff errors to produce a negative townsend(...)
        # It's faster to correct this than add an abs(...) below
        eta = convert(T, -alpha_tbl.ginv(minimum(alpha_tbl.gy)) * (1 + 1e-7))
        new{T, typeof(alpha_tbl),
            typeof(mu_tbl), typeof(dif_tbl)}(eta, alpha_tbl, mu_tbl, dif_tbl)
    end
end

@inline mobility(m::CWITransportModel, eabs, p=nothing) = m.mu(eabs)
@inline diffusion(m::CWITransportModel, eabs, p=nothing) = m.dif(eabs)
@inline nettownsend(m::CWITransportModel, eabs, p=nothing) = m.alpha(eabs)
@inline attachment(m::CWITransportModel, eabs, p=nothing) = m.eta
@inline townsend(m::CWITransportModel, eabs, p=nothing) = m.alpha(eabs) + m.eta
@inline netionization(m::CWITransportModel, eabs, p=nothing) = nettownsend(m, eabs) * eabs * m.mu(eabs)


########################################
# Transport model based on swarm data
########################################
struct TransportLookup{L <: LookupTable} <: AbstractTransportModel
    lookup::L
    mobility_index::Int
    diffusion_index::Int
    townsend_index::Int
    townsend_attachment_index::Int

    function TransportLookup(lookup; mobility_index, 
                             diffusion_index, 
                             townsend_index, 
                             townsend_attachment_index)
        new{typeof(lookup)}(lookup,
                            lookupindex(lookup, mobility_index), 
                            lookupindex(lookup, diffusion_index), 
                            lookupindex(lookup, townsend_index), 
                            lookupindex(lookup, townsend_attachment_index))
    end
end



@inline mobility(m::TransportLookup, eabs, p=nothing) = m.lookup(eabs, m.mobility_index)
@inline diffusion(m::TransportLookup, eabs, p=nothing) = m.lookup(eabs, m.diffusion_index)
@inline nettownsend(m::TransportLookup, eabs, p=nothing) = townsend(m, eabs, p) - attachment(m, eabs, p)
@inline attachment(m::TransportLookup, eabs, p=nothing) = m.lookup(eabs, m.attachment_index)
@inline townsend(m::TransportLookup, eabs, p=nothing) = m.lookup(eabs, m.townsend_index)
@inline netionization(m::TransportLookup, eabs, p=nothing) = nettownsend(m, eabs) * eabs * mobility(m, eabs)
