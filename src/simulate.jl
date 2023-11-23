#= 
Code to process input data and start a simulation.
=#

const MINIMAL_TRANSPORT_MODEL = MinimalTransportModel(;μ = 0.0372193,
                                                      D = 0.18,
                                                      αi = 433200.0,
                                                      Ei = 2e7,
                                                      αa = 2000.0,
                                                      Ea = 3e6)
const MINIMAL_CHEMICAL_MODEL = NetIonization(MINIMAL_TRANSPORT_MODEL)
const DATA_PATH = normpath(joinpath(@__DIR__, "../data"))
const STP_AIR_DENSITY = co.nair
const SYNTH_AIR_COMPOSITION = Dict("O2" => 0.2, "N2" => 0.8)
const Td = co.Td
const TOWNSEND_STP_AIR = co.Td * STP_AIR_DENSITY


@kwdef mutable struct InputParameters{T}
    "Name for the simulation."
    name::String = "noname-" * String(rand('A':'Z', 3)) * Dates.format(now(), "yyyymmdd_HHmmss")
    
    "Number of dimensions (2 or 3)"
    D::Int = 2
    
    "Block size"
    M = 8
    
    "Root-level block dimensions"
    rootsize = (1, 1)
    
    "Size of each block at root level"
    L::T = 1.25e-2
    
    "Absolute maximum level.  This is a hard limit; not very relevant."
    maxlevel::Int = 16
    
    "Boundary conditions for the Poisson equation."
    poisson_boundary = nothing
    
    "Use free boundary conditions?"
    freebnd::Bool = false
    
    # Initial conditions for electrons and ions
    "Peak of the electron Gaussian density"
    N0_e::T = 5e18

    "Width of the electron Gaussian density"
    w_e::T = 4e-4

    "Location of the electron Gaussian peak"
    z0_e::T = 0.01

    "Electron background density"
    nbg_e::T = 1e13

    "Whether to use a extended Gaussian (needle) and in what z-direction"
    extend_e::Int = 0
    
    "Peak of the ion Gaussian density"
    N0_i::T = N0_e

    "Width of the ion Gaussian density"
    w_i::T = w_e

    "Location of the ion Gaussian peak"
    z0_i::T = z0_e

    "Ion background density"
    nbg_i::T = nbg_e

    "Whether to use a extended Gaussian (needle) and in what z-direction"
    extend_i::Int = extend_e
    
    """
    Initial conditions.  Any function works but using some predefined types the code
    is generic for 2 or 3d.
    """
    initial_conditions = [1 => Background(nbg_e) + Gaussian(;A=N0_e, w=w_e, z0=z0_e,
                                                            extend=extend_e),
                          2 => Background(nbg_i) + Gaussian(;A=N0_i, w=w_i, z0=z0_i,
                                                            extend=extend_i)]
    
    
    "Background field."
    eb = @SVector([zero(T), T(-18.75e3 / 1.25e-2)])
    
    "Photoionization model."
    phmodel = empty_photoionization(T)
    
    "Transport model."
    trans = MINIMAL_TRANSPORT_MODEL
    
    "Chemical model."
    chem = MINIMAL_CHEMICAL_MODEL
    
    "Gas density model."
    dens = TrivialDensityScaling()
    
    "How many iterations between each recomputing the mesh?"
    refine_every::Int = 2
    
    "If not nothing, ignore the other refinement parameters and use this as refinement crit."
    refinement = nothing
    
    "Refinement persistence (s)."
    refine_persistence::T = 2e-10
    
    "Final time of the density-based refinement used for initial conditions."
    refine_density_upto::T = 4e-9
    
    "Refine up to this value in the density criterium."
    refine_density_h::T = 2e-4
    
    "Refine where density is above this in the density criterium."
    refine_density_value::T = 1e17
    
    "Do not derefine if the resulting spacing is larger than this."
    derefine_max_h::T = 20e-6
    
    "Do not refine finer than this h."
    refine_min_h::T = 3e-6
    
    "The c0 parameter for the Teunissen refinement criterium."
    refine_teunissen_c0::T = 0.5

    "The c1 parameter for the Teunissen refinement criterium."
    refine_teunissen_c1::T = 1.2
    
    "A refinement criterium based on the laplacian."
    refine_laplacian_alpha = nothing
    
    "The type of flux scheme (:koren / :weno)"
    flux_scheme::Symbol = :koren
    
    "Use full multigrid for the Poisson equation?"
    poisson_fmg::Bool = false
    
    "Number of Poisson iterations"
    poisson_iter::Int = 2
    
    "(up, down, top) iterations in V cycles in the Poisson solver."
    poisson_level_iter::NTuple{3, Int} = (2, 2, 4)
    
    "Save status?"
    save::Bool = true
    
    "A safety factor for time integration."
    dt_safety_factor::T = 0.8
    
    "Output times."
    output = 0:1e-9:16e-9
    
    "Final time."
    tend::T = output[end]
    
    # Storage mode:
    #  :contiguous is reasonably fast
    #  :vector can be faster for 3d computations but compilation may also be very long
    #   (hours perhaps).
    # Use :contiguous unless you expect your simulation to lasts very long
    # (from several hours). But note also that with :vector M cannot be too large or
    # you will die waiting for compilation.
    "Storage mode."
    storage::Symbol = :contiguous
    
    "Order of the poisson equation (2 and 4 are allowed)."
    poisson_order::Int = 2
    
    "Where to write output data"
    outfolder=expanduser("$(name)")

    "Functions to call at output snapshots."
    output_callbacks=()
end

"""
Start a simulation with the given parameters.

The parameters a received in `input` and possibly updated with additional keyword arguments.
Generally `input` should be of type `InputParameters` but you can use any type woth the appropriate
fields.
"""
function simulate(input=NamedTuple(); kw...)
    params = merge(ntfromstruct(input), kw)

    # Convert to InputParameters to enforce types and defaults
    local params1
    try
        params1 = InputParameters(params...)
    catch e
        @error "Error in the input parameters (see backtrace)"
        rethrow(e)
    end

    with_logger(TerminalLogger(meta_formatter=metafmt)) do
        _simulate(params1)
    end
end


function _simulate(input::InputParameters{T}) where T
    (;D, M, L) = input
    Polyester.reset_threads!()

    if !isdir(input.outfolder)
        mkdir(input.outfolder)
        @info "$(input.outfolder) created"
    else
        @warn "$(input.outfolder) already exists and output data may overwrite exisiting files."
    end


    # Set flux scheme Koren / WENO
    fluxschem = (input.flux_scheme == :koren ? FluxSchemeKoren() :
        input.flux_scheme == :weno ? FluxSchemeWENO() :
        throw(ArgumentError("Unknown flux scheme $(input.flux_scheme)")))
    
    # Koren needs 2 ghosts; WENO needs 4
    G = needghost(fluxschem)

    # Grid size at top level
    h = L / M
    derefine_minlevel = levelabove(input.derefine_max_h, h)
    refine_maxlevel = levelbelow(input.refine_min_h, h)
    tree = Tree(D, CartesianIndices(ntuple(i -> input.rootsize[i], Val(D))), input.maxlevel)

    # Boundary conditions for the Poisson equation
    pbc = (!isnothing(input.poisson_boundary) ?
           input.poisson_boundary : (D == 2 ?
                                     boundaryconditions(((1, -1), (-1, -1))) :
                                     boundaryconditions(((-1, -1), (-1, -1), (-1, -1)))))
        

    # Boundary conditions for the fluids
    fbc = (D == 2 ?
        boundaryconditions(((1, 1), (1, 1))) :
        boundaryconditions(((1, 1), (1, 1), (1, 1))))

    # Geometry of the problem and discretization for the Poisson equation.
    # CylindricalGeometry{d} indicates that d is treated as the radial coordinate
    geom = (D == 2 ? CylindricalGeometry{1}() : CartesianGeometry())    
    lpl = LaplacianDiscretization{D, input.poisson_order}()

    stencil = (input.poisson_order == 2 ? StarStencil{D}() :
        input.poisson_order == 4 ? BoxStencil{D}() :
        throw(ArgumentError("poisson_order = $(input.poisson_order) not allowed")))
        
    # The data struct that contains all streamer fields
    fields = StreamerFields(T, nspecies(input.chem), length(input.phmodel), D, M, G, Val(input.storage))
    freebcinst = input.freebnd ? FreeBC(geom, fields.u, refine_maxlevel, input.rootsize, h) : nothing
    
    ###
    #  REFINEMENT CRITERIUM
    ###
    if isnothing(input.refinement)
        # The Teunissen refinement criterium based on the electric field
        tref = TeunissenRef(fields.eabs, input.trans, input.refine_teunissen_c0,
                            input.refine_teunissen_c1, input.dens)
    
        if !isnothing(input.refine_laplacian_alpha)
            tref = AndRef(tref, LaplacianRef(input.refine_laplacian_alpha, fields.n[1],
                                             lpl, geom))
        end

        # A refinement creiterium based on the density of some species (here ions)    
        nref = DensityRef(fields.n[2], input.refine_density_value, input.refine_density_h)
        
        # We combine the previous criteria for a final one, including a persistence and
        # a time-limited version of the initial criterium.
        ref = PersistingRef(input.refine_persistence,
                            AndRef(TimeLimitedRef(zero(T), input.refine_density_upto, nref),
                                   tref))

    else
        ref = input.refinement
        nref = input.refinement
    end

    # Allow users to give eb as Vector. Performance impact sould be negligible as everything is done
    # below function boundaries.
    if input.eb isa AbstractVector
        eb = SVector{length(input.eb)}(input.eb)
    else
        eb = input.eb
    end
    
    # Set the streamer configuration
    conf = StreamerConf(h, eb, geom, fbc, pbc, lpl, freebcinst, input.trans, fluxschem,
                        input.chem, input.phmodel, input.dens, ref, stencil, input.dt_safety_factor,
                        input.poisson_fmg, input.poisson_iter, input.poisson_level_iter)
    
    # Start with a full tree up to level 2
    populate!(tree, 2)
    newblocks!(fields, nblocks(tree))
    newblocks!(freebcinst, nblocks(tree))

    conn = initial_conditions!(fields, conf, tree, nref, input.initial_conditions,
                               minlevel=2, maxlevel=refine_maxlevel)
    
    @info "Number of blocks after `initial_conditions!`" nblocks=nblocks(tree)
    
    # We store the location and value of the max. field for all output times.
    az = T[]
    at = T[]
    aemax = T[]

    function _report(t, conf, fields, tree, conn)
        (emax, r) = findmaxloc(fields.eabs, tree, h)
        
        zemax = r[end]
        @info "Snapshot" t zemax emax min_h=(h / 2^(findlast(!isempty, tree) - 1))
        
        push!(az, r[end])
        push!(at, t)
        push!(aemax, emax)
    end

    isave = 0
    function _save(t, conf, fields, tree, conn, label=nothing)
        if isnothing(label)
            label = format("{:05d}", isave)
            isave += 1
        end
        fname = joinpath(input.outfolder, label * ".jld2")
        jldsave(fname, true; fields, conf, tree, conn, t)
        @info "snapshot created" fname
    end
    _save_err(t, conf, fields, tree, conn) = _save(t, conf, fields, tree, conn, "error")
    
    output_callbacks = (input.output_callbacks..., _report,)
    onerror = ()

    if input.save
        output_callbacks = (output_callbacks..., _save)
        onerror = (onerror..., _save_err,)
    end
    

    run!(fields, conf, tree, conn, input.tend; progress_every=10, input.output,
         derefine_minlevel,
         refine_maxlevel,
         output_callbacks, onerror)

    return NamedTuple(Base.@locals)
end
