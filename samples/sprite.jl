"""
Simulate a funnel-like sprite similar to the one in [Malagón-Romero et al. 2020][1].

---
[1]: <https://doi.org/10.1029/2019GL085776> Malagón-Romero, A., Teunissen, J., Stenbaek-Nielsen, H. C.,
McHarg, M. G., Ebert, U., & Luque, A. (2020). On the emergence mechanism of carrot sprites. Geophysical Research Letters, 47, e2019GL085776.
"""
module Simulation

using Steamroll: InputParameters, simulate, @react_str
import Steamroll as sr

function setup()
    T = Float64
    name = splitext(@__FILE__)[1]
    zbase = 40e3
    ztop = 80e3
    hscale = 7.2e3
    
    ##
    # DEFINE MODELS
    ##
    lxfile = joinpath(sr.DATA_PATH, "LxCat_Phelps_20230914.txt")
    air_dens = sr.STP_AIR_DENSITY * exp(-zbase / hscale)
    composition = sr.SYNTH_AIR_COMPOSITION
    
    ## DENSITY SCALING
    dens = sr.ExpDensityScaling(hscale, 0.0)
    
    ## TRANSPORT MODEL
    clamp_mobility = (0.0, Inf)
    lx = sr.LxCatSwarmData.load(lxfile)
    
    lx.data.eabs = lx.data.en * sr.Td * air_dens
    
    # Rescale columns with air_dens
    lx.data.diffusion           = @. lx.data.A6 / air_dens
    lx.data.mobility            = @. lx.data.A2 / air_dens    
    lx.data.townsend            = @. air_dens * lx.data.A18
    lx.data.townsend_attachment = @. air_dens * lx.data.A19
    clamp!(lx.data.mobility, clamp_mobility...)
    
    lookup = sr.loadtable(eachcol(lx.data), xcol=:eabs,
                          ycols=[:mobility, :diffusion, :townsend, :townsend_attachment],
                          resample_into=2048)
    
    trans = sr.TransportLookup(lookup;
                               mobility_index=:mobility,
                               diffusion_index=:diffusion,
                               townsend_index=:townsend,
                               townsend_attachment_index=:townsend_attachment)
    
    ## CHEMICAL MODEL
    lx = sr.LxCatSwarmData.load(lxfile)
    
    lx.data.eabs = lx.data.en * sr.Td * air_dens
    
    lookup = sr.loadtable(eachcol(lx.data), xcol=:eabs,
                          ycols=[:C25, :C26, :C27, :C28, :C43],
                          resample_into=2048)
    
    chem = sr.ReactionSet(
        :pre =>
            [
                react"e + O2 -> 2 * e + O2+" => sr.RateLookup(lookup, :C43),
                react"e + N2 -> 2 * e + N2+" => sr.RateLookup(lookup, :C25),
                react"e + N2 -> 2 * e + N2+" => sr.RateLookup(lookup, :C26)],
        :post =>
            [
                react"e + O2 -> O + O-"      => sr.RateLookup(lookup, :C28),  
                react"e + O2 + Q -> O2- + Q" => sr.RateLookup(lookup, :C27)],
        
        # Altitude-dependent densities. x[end] is the z-component both in 2d and 3d.
        fix = (:N2 => x -> air_dens * composition["N2"] * exp(-x[end] / hscale),
               :O2 => x -> air_dens * composition["O2"] * exp(-x[end] / hscale),
               :M => x -> air_dens * exp(-x[end] / hscale),
               
               # normalization of lxcat.
               :Q => x -> air_dens / 1e6 * exp(-x[end] / hscale),
               :O => 0.0)) |> sr.Chemise
    
    
    ## INITIAL CONDITIONS
    a0 = 400.0
    d0 = 10e3
    hv = 60e3
    hb = 50e3
    
    a(z1) = max(a0, (d0 / 2) * (hv - z1) / (hv - hb))
    n(z1) = 2e11 * exp(-(z1 - 60e3) / 7.2e3)
    
    f = (sr.Funnel(n0 = z -> n(z + zbase), a = z -> a(z + zbase), a0=a0, z0=hb - zbase, z1=ztop - zbase)
         + sr.Cylinder(func = z -> 1e-2 * 1e6 * exp((z + zbase - 60e3) / 2.86e3),
                       R = 200e3))
    
    
    input = InputParameters{T}(
        ;
        name=name,
        trans=trans,
        chem=chem,
        dens=dens,
        
        rootsize=(1, 1),
        L = ztop - zbase,    
        freebnd = true,
        
        # Background field
        eb = [0.0, -72.38],
        
        initial_conditions = [1 => f, 2 => f],
        
        phmodel = sr.empty_photoionization(T),
        
        # The parameters for the Teunissen refinement criterium
        refine_teunissen_c0=0.7,
        refine_teunissen_c1=1.2,
        refine_persistence=4e-10,
        refine_density_h=10,
        refine_density_value=1e8,
        derefine_max_h=50,
        refine_min_h=5,
        
        poisson_fmg=false,
        poisson_iter=3,
        poisson_level_iter=(4, 4, 4),
        
        # Output times
        output=0:5e-4:100e-3)

    return input
end

const input = setup()

# Not run when we just import the module
run() = simulate(input)
end

if abspath(PROGRAM_FILE) == @__FILE__
    printstyled("This simulation has the following description:\n\n", color=:light_black)
    display(@doc(Simulation))
    println()
    Simulation.run()
end
