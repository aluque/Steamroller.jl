# bagheri.jl : Wed Sep 13 14:05:24 2023
"""
# Bagheri

This is an example code to run streamer 2d simulations.  By default it runs under the conditions
of Bagheri 2018, case 1.  By changing the parameters passed to main it is trivial to run case 2.
## Running the code

```julia
julia> includet("bagheri.jl")     # Needs Revise.jl
julia> Bagheri.main()
```
"""
module Bagheri

using StaticArrays
#using ProgressMeter
using Logging
using Dates
using Formatting
using Printf
using TerminalLoggers
using ProgressLogging
import Steamroller as sr
import SteamrollerPlot as srplt
import PyPlot as plt
using DataFrames
using CSV
using Polyester

const co = sr.co

function main(;kw...)
    with_logger(TerminalLogger(meta_formatter=sr.metafmt)) do
        _main(;kw...)
    end
end

case1(;kw...) = main(;kw...)
case2(;kw...) = main(nbg=1e9, output=0:1e-9:24.01e-9, tend=24.01e-9, refine_persistence=5e-12,
                     v=0.03e7; kw...)
case3(;kw...) = main(nbg=1e9, output=0:1e-9:15.01e-9, tend=15.01e-9, refine_persistence=5e-12,
                     phmodel=sr.bourdon3(), v=0.06e7; kw...)
case1_3d(;kw...) = main(;D=3, rootsize=(2, 2, 1),
                        eb = @SVector([0.0, 0.0, -18.75e3 / 1.25e-2]),
)

function _main(;
               # The data type for floating point operations.  In general changing this to
               # Float32 does not provide much improvement in performance but you may consider
               # it in case of very large simulations where memory space is a problem.
               T=Float64,
               
               # Number of dimensions
               D=2,
               
               # Heavy species
               H=1,

               # Block size
               M = 8,
               
               # Root-level block dimensions
               rootsize = (1, 1),
               
               # Size of each block
               L = T(1.25e-2),
               
               # Absolute maximum level.  This is a hard limit; not very relevant.
               maxlevel = 16,
               
               # Final time
               tend=T(16e-9),
               
               # Initial conditions.
               N0  = T(5e18),       # Peak of the gaussian ion density
               σ   = T(4e-4),       # Width of the gaussian
               z0  = T(0.01),       # Location of the gaussian peak
               nbg = T(1e13),       # Background density

               initial_conditions = (D == 2 ?
                                     [1 => (r, z) -> nbg,                              # electrons
                                      2 => (r, z) -> nbg + gaussian(N0, z, r, z0, σ)] :# ions
                                     [1 => (x, y, z) -> nbg,
                                      2 => (x, y, z) -> nbg + gaussian(N0, x, y, z, L, L, z0, σ)]),
    

               # Background field
               eb = @SVector([zero(T), T(-18.75e3 / 1.25e-2)]),

               # Photoionization model
               phmodel = sr.empty_photoionization(T),
               
               
               # How many iterations between each recomputing the mesh?
               refine_every=2,
                              
               # To improve refinement in the inside of the space-charge layer we let the refinement
               # stay for a while.  This is this persistence time, in seconds.
               refine_persistence=T(2e-10),
               
               # End of density-based refinement
               refine_density_upto=T(4e-9),

               # Refine up to this value in the density criterium
               refine_density_h=T(1e-5),

               # Refine where density is above this in the density criterium
               refine_density_value=T(1e16),
               
               # Do not derefine if the resulting spacing is larger than this
               derefine_max_h=T(20e-6),
               
               # Do not refine below this level
               refine_min_h=T(3e-6),

               # The parameters for the Teunissen refinement criterium
               refine_teunissen_c0=T(1.0),
               refine_teunissen_c1=T(1.2),
               
               # The type of flux scheme
               flux_scheme=:koren,

               # Plot the evolution of the streamer?
               plot=false,
               
               # A safety factor for time integration
               dt_safety_factor=T(0.8),
               
               # Output times
               output=0:1e-9:16e-9,

               # Approximate velocity of the streamer (only for comparison with Bagheri 2018 data)
               v=0.05e7,
               
               # Storage mode:
               #  :contiguous is reasonably fast
               #  :vector can be faster for 3d computations but compilation may also be very long
               #   (hours perhaps).
               storage=:contiguous,

               # Order of the poisson equation (2 and 4 are allowed).
               poisson_order=2
               )
    
    Polyester.reset_threads!()
    
    fluxschem = flux_scheme == :koren ? sr.FluxSchemeKoren() :
        flux_scheme == :weno ? sr.FluxSchemeWENO() :
        @error "Unknown flux scheme $flux_scheme"

    G = sr.needghost(fluxschem)
    
    # This makes block size of 1 cm.
    h = L / M
    derefine_minlevel = sr.levelabove(derefine_max_h, h)
    refine_maxlevel = sr.levelbelow(refine_min_h, h)
        
    fields = sr.StreamerFields(T, 2, length(phmodel), D, M, G, Val(storage))
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
    geom = (D == 2 ? sr.CylindricalGeometry{1}() : sr.CartesianGeometry())    
    lpl = sr.LaplacianDiscretization{D, poisson_order}()

    stencil = poisson_order == 2 ? sr.StarStencil{D}() :
        poisson_order == 4 ? sr.BoxStencil{D}() : @error "poisson_order = $poisson_order not allowed"

    
    trans = sr.BagheriTransportModel()

    # Chemical model
    chem = sr.NetIonization(trans)
    
    # The Teunissen refinement criterium based on the electric field
    tref = sr.TeunissenRef(fields.eabs, trans, refine_teunissen_c0, refine_teunissen_c1)
    
    # A refinement creiterium based on the density of some species (here ions)    
    nref = sr.DensityRef(fields.n[2], refine_density_value, refine_density_h)

    # We combine the previous criteria for a final one, including a persistence and a time-limited
    # version of the initial criterium.
    ref = sr.PersistingRef(refine_persistence,
                           sr.AndRef(sr.TimeLimitedRef(zero(T), refine_density_upto, nref), tref))


    # Set the streamer configuration
    conf = sr.StreamerConf(h, eb, geom, fbc, pbc, lpl, trans, fluxschem, chem,
                           phmodel, sr.TrivialDensityScaling(), ref, stencil, dt_safety_factor)
    
    # Start with a full tree up to level 3
    sr.populate!(tree, 3)
    sr.newblocks!(fields, sr.nblocks(tree))

    conn = sr.initial_conditions!(fields, conf, tree, nref, initial_conditions,
                                  minlevel=derefine_minlevel, maxlevel=refine_maxlevel)
    
    @info "Number of blocks after initial conditions!" nblocks=sr.nblocks(tree)
    
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
        if plot
            plt.clf()
            srplt.scalartreeplot(fields.eabs, tree, h, boundaries=true)
            plt.colorbar()
            
            # This gc here seems to prevent a nasty segfault bug in PyCall/PyPlot
            GC.gc()
        end
    end
    
    sr.run!(fields, conf, tree, conn, tend; output, derefine_minlevel, refine_maxlevel,
            output_callbacks=(_report, _plot))
    df = DataFrame(t=at, z=az, emax=aemax)
    show(resultdf(df; v))

    return NamedTuple(Base.@locals)
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
    Bagheri.main()
end

