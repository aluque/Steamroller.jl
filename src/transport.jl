#All around transport models (including impact ionization and attachment).

abstract type AbstractTransportModel end

"""
The transport model proposed in Bagheri et al. 2018.
"""
Base.@kwdef struct BagheriTransportModel{T} <: AbstractTransportModel
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
Velocity in transport model `m` for an electric field `eabs`.
"""
@inline mobility(m::BagheriTransportModel, eabs) = @fastmath m.μ0 * eabs^m.eμ
#@inline mobility(m::BagheriTransportModel, eabs) = 0.0372193


"""
Diffusion rate in transport model `m` for an electric field `eabs`.
"""
@inline diffusion(m::BagheriTransportModel, eabs) = @fastmath m.D0 * eabs^m.eD


"""
Townsend coefficient (excluding attachment) in transport model `m` for an electric field `eabs`.
"""
@inline function townsend(m::BagheriTransportModel, eabs)
    (;α0, α1, eα, mα, η) = m

    @fastmath (α0 + α1 / eabs^eα) * exp(mα / eabs)
end

"""
Townsend attachment coefficient in transport model `m` for an electric field `eabs`.
"""
@inline attachment(m::BagheriTransportModel, eabs) = m.η

"""
Townsend coefficient in transport model `m` for an electric field `eabs`.
"""
@inline function nettownsend(m::BagheriTransportModel, eabs)
    @fastmath townsend(m, eabs) - attachment(m, eabs)
end


"""
Net ionization in transport model `m` for an electric field `eabs`.
"""
@inline function netionization(m::BagheriTransportModel, eabs)
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

@inline mobility(m::CWITransportModel, eabs) = m.mu(eabs)
@inline diffusion(m::CWITransportModel, eabs) = m.dif(eabs)
@inline nettownsend(m::CWITransportModel, eabs) = m.alpha(eabs)
@inline attachment(m::CWITransportModel, eabs) = m.eta
@inline townsend(m::CWITransportModel, eabs) = m.alpha(eabs) + m.eta
@inline netionization(m::CWITransportModel, eabs) = nettownsend(m, eabs) * eabs * m.mu(eabs)
