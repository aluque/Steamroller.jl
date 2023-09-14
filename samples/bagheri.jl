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
import Steamroll as sr
import SteamrollPlot as srplt
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
                     phmodel=bourdon3(), v=0.06e7; kw...)

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
               
               # Absolute maximum level.  This is a hard limit; not very relevant.
               maxlevel = 16,
               
               # Final time
               tend=T(16e-9),
               
               # Initial conditions.
               N0  = T(5e18),       # Peak of the gaussian ion density
               σ   = T(4e-4),       # Width of the gaussian
               z0  = T(0.01),       # Location of the gaussian peak
               nbg = T(1e13),       # Background density

               # Background field
               eb = @SVector([zero(T), T(-18.75e3 / 1.25e-2)]),

               # Photoionization model
               phmodel = empty_photoionization(T),
               
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
               
               # Do not derefine below this level
               refine_min_h=T(3e-6),

               # The parameters for the Teunissen refinement criterium
               refine_teunissen_c0=T(0.5),
               refine_teunissen_c1=T(1.2),
               
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
    
    # This makes block size of 1 cm.
    h = T(1.25e-2) / M
    derefine_minlevel = sr.levelabove(derefine_max_h, h)
    refine_maxlevel = sr.levelbelow(refine_min_h, h)
        
    fields = sr.StreamerFields(T, 2, length(phmodel), D, M, H, Val(storage))
    tree = sr.Tree(D, CartesianIndices(ntuple(i -> rootsize[i], Val(D))), maxlevel)

    # Boundary conditions for the Poisson equation
    pbc = sr.boundaryconditions(((1, -1), (-1, -1)))

    # Boundary conditions for the fluids
    fbc = sr.boundaryconditions(((1, 1), (1, 1)))

    # Geometry of the problem and discretization for the Poisson equation.
    geom = sr.CylindricalGeometry{1}()
    lpl = sr.LaplacianDiscretization{2, poisson_order}()

    stencil = poisson_order == 2 ? sr.StarStencil{2}() :
        poisson_order == 4 ? sr.BoxStencil{2}() : @error "poisson_order = $poisson_order not allowed"

    
    trans = sr.BagheriTransportModel()
    # To use lookup tables:
    # trans = sr.CWITransportModel{T}(
    #     joinpath.(@__DIR__, "data", ["cwi_alpha.txt", "cwi_mu.txt", "cwi_dif.txt"])...)
    
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
    conf = sr.StreamerConf(h, eb, geom, fbc, pbc, lpl, trans, chem,
                           phmodel, ref, stencil, dt_safety_factor)
    
    # Start with a full tree up to level 3
    sr.populate!(tree, 3)
    sr.newblocks!(fields, sr.nblocks(tree))

    conditions = [1 => (r, z) -> nbg,                            # electrons
                  2 => (r, z) -> nbg + gaussian(N0, z, r, z0, σ) # ions
                  ]
    conn = sr.initial_conditions!(fields, conf, tree, nref, conditions, minlevel=3, maxlevel=refine_maxlevel)
    
    @info "Number of blocks after initial conditions!" nblocks=sr.nblocks(tree)
    
    # Measure times
    elapsed_step = 0.0
    elapsed_refine = 0.0
    elapsed_connectivity = 0.0
    
    t = zero(T)
    dt = zero(T)

    output = map(x->convert(T, x), output)

    # We store the location and value of the max. field for all output times.
    az = T[]
    at = T[]
    aemax = T[]

    logger = Logging.current_logger()
    local msg
    min_dt = 1e-16
    
    @withprogress begin
        iter = 0
        while t < tend
            elapsed_step += @elapsed (t, dt) = sr.step!(fields, conf, tree, conn, t,
                                                        get(output, 1, convert(T, Inf)),
                                                        Val(:ssprk3))
            
            if !isempty(output) && isapprox(t, first(output))
                (emax, r) = findmax(fields.eabs, tree, h)
                
                zemax = r[end]
                @info "Snapshot time reached" t zemax emax

                push!(az, r[end])
                push!(at, t)
                push!(aemax, emax)
                
                if plot
                    #plt.figure(1)
                    plt.clf()
                    srplt.scalartreeplot(fields.eabs, tree, h, boundaries=true)
                    # plt.xlim([0, 2e-3])
                    # plt.ylim([0.000, 0.011])
                    plt.colorbar()

                    # This gc here seems to prevent a nasty segfault bug in PyCall/PyPlot
                    GC.gc()
                end
                popfirst!(output)
            end

            if t > 0 && dt < min_dt
                @warn "dt is below the minimal allowed min_dt.  Stopping the iterations here." dt min_dt
                break
            end
            
            if (iter % refine_every) == 0                
                elapsed_refine += @elapsed sr.refine!(fields, conf, tree, conn, t, dt;
                                                      minlevel=derefine_minlevel,
                                                      maxlevel=refine_maxlevel)
                # minlevel was 8
                elapsed_connectivity += @elapsed conn = sr.connectivity(tree, stencil)
            end

            local frac = (t / tend)^2 
            if (iter % 50) == 0
                @logprogress frac
                msg = join(map(x -> @sprintf("%30s = %-30s", string(first(x)), repr(last(x))), 
                               Pair{Symbol, Any}[:t => t,
                                                 :dt => dt,
                                                 :iter => iter,
                                                 :max_level => findlast(!isempty, tree),
                                                 :min_h => h / 2^(findlast(!isempty, tree) - 1),
                                                 :nblocks => sr.nblocks(tree),
                                                 :elapsed_step => elapsed_step,
                                                 :elapsed_refine => elapsed_refine,
                                                 :elapsed_connectivity => elapsed_connectivity]), "\n")
                io = IOBuffer()
                printstyled(IOContext(io, :color=>true), msg, color=:blue)
                push!(Logging.current_logger().sticky_messages, :vars=>String(take!(io)))
            end
            iter += 1
        end
    end
    @info "\n```\n$msg\n```"
    
    empty!(Logging.current_logger().sticky_messages)
    df = DataFrame(t=at, z=az, emax=aemax)
    CSV.write("streamer2d.csv", df)
    @info "Location of the streamer tip" resultdf(df; v)
    return NamedTuple(Base.@locals)
end


"""
Construct the bourdon 2/3-term model as described by Bagheri 2018.
"""
function bourdon(T, A, λ)
    # ξB νu/νi
    ξ´ = 0.075
    pO2 = 150 * co.torr
    p = 750 * co.torr
    pq = 40 * co.milli * co.bar


    # See A.9 for the multiplicative factors.  Note also the unit conversion.
    a = @. A * pO2^2 * (pq / (p + pq)) * ξ´ * co.centi^-2 * co.torr^-2
    k = @. pO2 * λ * co.centi^-1 * co.torr^-1

    term = sr.PhotoionizationTerm{T}.(a, k .^ 2)
    bc = sr.boundaryconditions(((1, 1), (-1, -1)))
    phmodel = sr.PhotoionizationModel{length(A), T, typeof(bc)}(term, 1, (1, 2), bc, 2)

    return phmodel
end

# Direct copy of table A3
bourdon3(T=Float64) = bourdon(T,
                              [1.986e-4, 0.0051, 0.4886],  # A
                              [0.0553, 0.1460, 0.8900])   # λ

"""
Construct an empty photo-ionization model.
"""
function empty_photoionization(T)    
    return sr.PhotoionizationModel{0, T, Nothing}(SVector{0, sr.PhotoionizationTerm{T}}(), 0, (0, 0), nothing, 0)    
end

L(z) = 1.25e-2 - z
DL(z::Real, t::Real; v = 0.05 * co.centi / co.nano) = L(z) - v * t
resultdf(x; v = 0.05 * co.centi / co.nano) = DataFrame(t=x.t, DL=DL.(x.z, x.t; v), L=L.(x.z), emax=x.emax)


function gaussian(a, z, r, z0, w)
    return a * exp(-(r^2 + (z - z0)^2) / w^2)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    Streamer2d.main()
end

