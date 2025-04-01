"""
Reproduce the results from [Malagon & Luque 2019][1].

---
[1]: <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL082063> "Malagón-Romero, A., & Luque, A. (2019). Spontaneous emergence of space stems ahead of negative
leaders in lightning and long sparks. Geophysical Research Letters, 46."
"""
module Simulation

using Steamroller: InputParameters, simulate, @react_str
import Steamroller as sr

function setup()
    T = Float64
    name = splitext(@__FILE__)[1]
    
    ##
    # DEFINE MODELS
    ##
    lxfile = joinpath(sr.DATA_PATH, "LxCat_Phelps_20230914.txt")
    air_dens = sr.STP_AIR_DENSITY
    composition = sr.SYNTH_AIR_COMPOSITION
    
    ## TRANSPORT MODEL
    clamp_mobility = (0.0, 0.1)
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
        
        fix = (:N2 => air_dens * composition["N2"],
               :O2 => air_dens * composition["O2"],
               :M => air_dens,
               
               # Normalization of 3-body attachment by bolsig
               :Q => air_dens / 1e6,
               
               # Disregard O
               :O => 0.0),
    ) |> sr.Chemise
    
    
    ## INITIAL CONDITIONS
    f = (sr.Background(1.0e15) +
        sr.Gaussian(;A=1e21, w=0.003 * sqrt(2),  z0=0.05, extend=-1) +
        sr.Gaussian(;A=1e21, w=0.0015 * sqrt(2), z0=0.061, extend=0))
    
    
    input = InputParameters{T}(
        ;
        name=name,
        trans=trans,
        chem=chem,
        rootsize=(1, 8),
        L=0.03,
        freebnd=true,
        
        # Background field
        eb = [0.0, -10e5],
        
        initial_conditions = [1 => f, 2 => f],
        
        phmodel = sr.empty_photoionization(Float64), #sr.bourdon3(Float64),
        
        refine_density_value=1e19,
        refine_density_upto=1e-9,
        
        # Do not derefine if the resulting spacing is larger than this
        derefine_max_h=4e-4,
        
        # The parameters for the Teunissen refinement criterium
        refine_teunissen_c0=0.8,
        refine_teunissen_c1=1.35,
        
        refine_persistence=1e-9,
        
        poisson_fmg=false,
        poisson_iter=3,
        poisson_level_iter=(4, 4, 4),
        
        # Output times
        output=0:1e-9:100e-9)
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
