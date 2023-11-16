# generic.jl : Fri Sep 15 15:28:50 2023
"""
# Generic
A more generic code reading from LxCat data

## Running the code

```julia
julia> includet("generic.jl")     # Needs Revise.jl
julia> Generic.main()
```
"""
module Generic

using StaticArrays
#using ProgressMeter
using Logging
using Dates
using Formatting
using Printf
using TerminalLoggers
using ProgressLogging
import Steamroll as sr
using Steamroll: @react_str
import SteamrollPlot as srplt
import PyPlot as plt
using DataFrames
using CSV
using Polyester
using JLD2

const co = sr.co

function main(;kw...)
    with_logger(TerminalLogger(meta_formatter=sr.metafmt)) do
        _main(;kw...)
    end
end

# Some examples
function testfreebc(;kw...)
    z0 = 0.4
    L = 0.2
    A = 1.0e10
    w = 0.06
    
    initial_conditions = [2 => sr.Gaussian(;A, w, z0, s=40) + sr.Background(1.0)]
    rootsize = (1, 4)
    H = L * rootsize[2]
    R = L * rootsize[1]

    z = LinRange(0, H, rootsize[2] * 2^9)
    u0 = zeros(size(z))
    k0 = (1/3) * w^3 * A * co.elementary_charge / co.epsilon_0
    for image in -100000:100000
        zimg = z0 + image * H
        #sgn = mod(fld(image, 2), 2) == 0 ? 1 : -1
        sgn = mod(image, 2) == 0 ? 1 : -1
        u0 .+= @. sgn * k0 / sqrt((z - zimg)^2 + R^2)
    end
    
    sim = main(;rootsize,
               eb = @SVector([0.0, 0.0]),
               L,
               refine_min_h = L / 8 / 2^2,
               refine_density_value=1e-9,
               refine_density_h=32 * 2e-4,
               refine_density_upto=5e-9,
               initial_conditions,
               kw...)
    
    return (;NamedTuple(Base.@locals)..., sim...)
end


function malagon(;kw...)
    f = (sr.Background(1.0e15) +
         sr.Gaussian(;A=1e21, w=0.003 * sqrt(2),  z0=0.05, extend=-1) +
         sr.Gaussian(;A=1e21, w=0.0015 * sqrt(2), z0=0.061, extend=0))
                         
    main(;rootsize=(1, 8),
         L=0.03,
         freebnd=true,
         
         # Background field
         eb = @SVector([0.0, -10e5]),

         initial_conditions = [1 => f, 2 => f],

         phmodel = sr.empty_photoionization(Float64), #sr.bourdon3(Float64),

         #phmodel = sr.bourdon3(Float64),
         clamp_mobility = (0.0, 0.1),

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
         output=0:1e-9:100e-9,
         save=true,
         outfolder=expanduser("~/data/steamroll/malagon-fbc-01"),
         kw...)
end

function _main(;
               # If you do not provide a name, invent one based on a random 3-letter string and
               # the current time.
               name="noname-" * String(rand('A':'Z', 3)) * Dates.format(now(), "yyyymmdd_HHmmss"),

               # The data type for floating point operations.  In general changing this to
               # Float32 does not provide much improvement in performance but you may consider
               # it in case of very large simulations where memory space is a problem.
               T=Float64,
               
               # Number of dimensions
               D=2,
               
               # Block size
               M = 8,
               
               # Root-level block dimensions
               rootsize = (1, 1),
               
               # Size of each block at root level
               L = T(1.25e-2),
               
               # Absolute maximum level.  This is a hard limit; not very relevant.
               maxlevel = 16,
               
               # Use free boundary conditions?
               freebnd = false,
                              
               # Initial conditions for electrons and ions
               N0_e  = 5e18,       # Peak of the gaussian ion density
               σ_e   = 4e-4,       # Width of the gaussian
               z0_e  = 0.01,       # Location of the gaussian peak
               nbg_e = 1e13,       # Background density
               extend_e = 0,
               
               N0_i  = N0_e,
               σ_i   = σ_e,
               z0_i  = z0_e,
               nbg_i = nbg_e,
               extend_i = 0,
               
               # Initial conditions.  Any function works but using some predefined types the code
               # is generic for 2 or 3d.
               initial_conditions = [1 => sr.Background(nbg_e) + sr.Gaussian(;A=N0_e, w=σ_e, z0=z0_e,
                                                                             extend=extend_e),
                                     2 => sr.Background(nbg_i) + sr.Gaussian(;A=N0_i, w=σ_i, z0=z0_i,
                                                                             extend=extend_i)],
               
               
               # Background field
               eb = @SVector([zero(T), T(-18.75e3 / 1.25e-2)]),
               
               # Photoionization model
               phmodel = sr.empty_photoionization(T),
               
               datapath = normpath(joinpath(@__DIR__, "../data")),
               lxfile = joinpath(datapath, "LxCat_Phelps_20230914.txt"),
               
               air_dens = co.nair,
               composition = Dict("O2" => 0.2, "N2" => 0.8),
               
               # Restrict mobility to be between these values
               clamp_mobility = (0.0, Inf),

               # Transport model
               trans = transport_model(lxfile, air_dens; clamp_mobility),

               # Chemical model
               chem = chemical_model(lxfile, air_dens, composition),

               # Allow ihomogeneous gas density.  Defaults to a homogeneous density.
               dens = sr.TrivialDensityScaling(),
               
               # How many iterations between each recomputing the mesh?
               refine_every=2,
               
               # If not nothing, ignore the other refinement parameters and use this as
               # refinement crit.
               refinement = nothing,

               # To improve refinement in the inside of the space-charge layer we let the refinement
               # stay for a while.  This is this persistence time, in seconds.
               refine_persistence=T(2e-10),
               
               # Final time of the density-based refinement used for initial conditions.
               refine_density_upto=T(4e-9),
               
               # Refine up to this value in the density criterium
               refine_density_h=T(2e-4),
               
               # Refine where density is above this in the density criterium
               refine_density_value=T(1e17),
               
               # Do not derefine if the resulting spacing is larger than this
               derefine_max_h=T(20e-6),
               
               # Do not refine finer than this h
               refine_min_h=T(3e-6),
               
               # The parameters for the Teunissen refinement criterium
               refine_teunissen_c0=T(0.5),
               refine_teunissen_c1=T(1.2),
               
               # A refinement criterium based on the laplacian.
               refine_laplacian_alpha=nothing,
               
               # The type of flux scheme
               flux_scheme=:koren,
               
               # Use full multigrid for the Poisson equation?
               poisson_fmg=false,

               # Number of Poisson iterations
               poisson_iter=2,
               
               # (up, down, top) iterations in V cycles in Poisson.
               poisson_level_iter=(2, 2, 4),

               # Plot the evolution of the streamer?
               plot=false,
               
               # Save status?
               save=true,
               
               # A safety factor for time integration
               dt_safety_factor=T(0.8),
               
               # Output times
               output=0:1e-9:16e-9,
               
               # Final time
               tend=output[end],

               # Approximate velocity of the streamer (only for comparison with Bagheri 2018 data)
               v=0.05e7,
               
               # Storage mode:
               #  :contiguous is reasonably fast
               #  :vector can be faster for 3d computations but compilation may also be very long
               #   (hours perhaps).
               # Use :contiguous unless you expect your simulation to lasts very long
               # (from several hours). But note also that with :vector M cannot be too large or
               # you will die waiting for compilation.
               storage=:contiguous,
               
               # Order of the poisson equation (2 and 4 are allowed).
               poisson_order=2,
               
               outfolder=expanduser("~/data/steamroll/$(name)")
               )
         
         
    Polyester.reset_threads!()

    # Set flux scheme Koren / WENO
    fluxschem = (flux_scheme == :koren ? sr.FluxSchemeKoren() :
                 flux_scheme == :weno ? sr.FluxSchemeWENO() :
                 throw(ArgumentError("Unknown flux scheme $flux_scheme")))
    
    # Koren needs 2 ghosts; WENO needs 4
    G = sr.needghost(fluxschem)

    # Grid size at top level
    h = L / M
    derefine_minlevel = sr.levelabove(derefine_max_h, h)
    refine_maxlevel = sr.levelbelow(refine_min_h, h)
        
    tree = sr.Tree(D, CartesianIndices(ntuple(i -> rootsize[i], Val(D))), maxlevel)

    # Boundary conditions for the Poisson equation
    pbc = (D == 2 ?
           sr.boundaryconditions(((1, -1), (-1, -1))) :
           sr.boundaryconditions(((-1, -1), (-1, -1), (-1, -1))))
        

    # Boundary conditions for the fluids
    fbc = (D == 2 ?
        sr.boundaryconditions(((1, 1), (1, 1))) :
        sr.boundaryconditions(((1, 1), (1, 1), (1, 1))))

    # Geometry of the problem and discretization for the Poisson equation.
    # CylindricalGeometry{d} indicates that d is treated as the radial coordinate
    geom = (D == 2 ? sr.CylindricalGeometry{1}() : sr.CartesianGeometry())    
    lpl = sr.LaplacianDiscretization{D, poisson_order}()

    stencil = (poisson_order == 2 ? sr.StarStencil{D}() :
               poisson_order == 4 ? sr.BoxStencil{D}() :
               throw(ArgumentError("poisson_order = $poisson_order not allowed")))
        
    # The data struct that contains all streamer fields
    fields = sr.StreamerFields(T, sr.nspecies(chem), length(phmodel), D, M, G, Val(storage))
    freebcinst = freebnd ? sr.FreeBC(geom, fields.u, refine_maxlevel, rootsize, h) : nothing
    
    ###
    #  REFINEMENT CRITERIUM
    ###
    if isnothing(refinement)
        # The Teunissen refinement criterium based on the electric field
        tref = sr.TeunissenRef(fields.eabs, trans, refine_teunissen_c0, refine_teunissen_c1)
    
        if !isnothing(refine_laplacian_alpha)
            tref = sr.AndRef(tref, sr.LaplacianRef(refine_laplacian_alpha, fields.n[1],
                                                   lpl, geom))
        end

        # A refinement creiterium based on the density of some species (here ions)    
        nref = sr.DensityRef(fields.n[2], refine_density_value, refine_density_h)
        
        # We combine the previous criteria for a final one, including a persistence and
        # a time-limited version of the initial criterium.
        ref = sr.PersistingRef(refine_persistence,
                               sr.AndRef(sr.TimeLimitedRef(zero(T), refine_density_upto, nref),
                                         tref))

    else
        ref = refinement
        nref = refinement
    end
    
    # Set the streamer configuration
    conf = sr.StreamerConf(h, eb, geom, fbc, pbc, lpl, freebcinst, trans, fluxschem, chem,
                           phmodel, dens, ref, stencil, dt_safety_factor,
                           poisson_fmg, poisson_iter, poisson_level_iter)
    
    # Start with a full tree up to level 2
    sr.populate!(tree, 2)
    sr.newblocks!(fields, sr.nblocks(tree))
    sr.newblocks!(freebcinst, sr.nblocks(tree))

    conn = sr.initial_conditions!(fields, conf, tree, nref, initial_conditions,
                                  minlevel=3, maxlevel=refine_maxlevel)
    
    @info "Number of blocks after `initial_conditions!`" nblocks=sr.nblocks(tree)
    
    # We store the location and value of the max. field for all output times.
    az = T[]
    at = T[]
    aemax = T[]

    function _report(t)
        (emax, r) = sr.findmaxloc(fields.eabs, tree, h)
        
        zemax = r[end]
        @info "Snapshot" t zemax emax min_h=(h / 2^(findlast(!isempty, tree) - 1))
        
        push!(az, r[end])
        push!(at, t)
        push!(aemax, emax)
    end

    function _plot(t)
        plt.clf()
        srplt.scalartreeplot(fields.eabs, tree, h, boundaries=true)
        plt.xlim([0, 3e-2])
        plt.ylim([0, 0.25])
        plt.colorbar()
        
        # This gc here seems to prevent a nasty segfault bug in PyCall/PyPlot
        GC.gc()
    end

    isave = 0
    function _save(t, label=nothing)
        if isnothing(label)
            label = format("{:05d}", isave)
            isave += 1
        end
        fname = joinpath(outfolder, label * ".jld2")
        jldsave(fname, true; fields, conf, tree, conn, t)
        @info "snapshot created" fname
    end
    _save_err(t) = _save(t, "error")
    
    output_callbacks = (_report,)
    onerror = ()

    if plot
        output_callbacks = (output_callbacks..., _plot)
    end

    if save
        output_callbacks = (output_callbacks..., _save)
        onerror = (onerror..., _save_err,)
    end

    
    sr.run!(fields, conf, tree, conn, tend; progress_every=10, output,
            derefine_minlevel,
            refine_maxlevel,
            output_callbacks, onerror)

    return NamedTuple(Base.@locals)
end

function transport_model(lxfile, air_dens; clamp_mobility=(0, Inf))
    lx = sr.LxCatSwarmData.load(lxfile)
    
    lx.data.eabs = lx.data.en * co.Td * air_dens

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
    return trans
end

function chemical_model(lxfile, air_dens, composition)
    lx = sr.LxCatSwarmData.load(lxfile)
    
    lx.data.eabs = lx.data.en * co.Td * air_dens
    
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
               :Q => air_dens / (co.centi^-3),

               # Disregard O
               :O => 0.0),
    ) |> sr.Chemise

    return chem
end

function net_ionization_chemical_model(lxfile, air_dens, composition)
    lx = sr.LxCatSwarmData.load(lxfile)
    
    lx.data.eabs = lx.data.en * co.Td * air_dens
    lx.data.ionization = @. (air_dens *
                             (composition["O2"] * lx.data.C43 +
                              composition["N2"] * (lx.data.C25 + lx.data.C26)))
    lx.data.attachment = @. (air_dens * composition["O2"] * (air_dens / (co.centi^-3) * lx.data.C27 +
                                                             lx.data.C28))
    lookup = sr.loadtable(eachcol(lx.data), xcol=:eabs,
                          ycols=[:ionization, :attachment],
                          resample_into=2048)

    chem = sr.NetIonizationLookup(;lookup, ionization_index=:ionization, attachment_index=:attachment)
end


L(z) = 1.25e-2 - z
DL(z::Real, t::Real; v = 0.05 * co.centi / co.nano) = L(z) - v * t
resultdf(x; v = 0.05 * co.centi / co.nano) = DataFrame(t=x.t, DL=DL.(x.z, x.t; v), L=L.(x.z), emax=x.emax)


function gaussian(a, z, r, z0, w)
    return a * exp(-(r^2 + (z - z0)^2) / w^2)
end

function gaussian(a, x, y, z, x0, y0, z0, w)
    return a * exp(-((x - x0)^2 + (y - y0)^2 + (z - z0)^2) / w^2)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    #Generic.main()
    Generic.malagon(;poisson_order=4)
end

