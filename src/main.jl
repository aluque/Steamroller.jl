#= 
Code to process input data and start a simulation.
=#

minimal_transport_model(::Type{T}=Float64) where T = MinimalTransportModel{T}(;μ = 0.0372193,
                                                                                 D = 0.18,
                                                                                 αi = 433200.0,
                                                                                 Ei = 2e7,
                                                                                 αa = 2000.0,
                                                                                 Ea = 3e6)
const MINIMAL_TRANSPORT_MODEL = minimal_transport_model()
const MINIMAL_CHEMICAL_MODEL = NetIonization(MINIMAL_TRANSPORT_MODEL)
const DATA_PATH = normpath(joinpath(@__DIR__, "../data"))
const STP_AIR_DENSITY = co.nair
const SYNTH_AIR_COMPOSITION = Dict("O2" => 0.2, "N2" => 0.8)
const Td = co.Td
const TOWNSEND_STP_AIR = co.Td * STP_AIR_DENSITY


@kwdef mutable struct InputParameters{T}
    "Number of dimensions (2 or 3)"
    D::Int = 2
    
    "Block size"
    M::Int = 8
    
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
    
    "Start the tree by fully populating it up to this level"
    init_populate_level = 2
    
    "Background field."
    eb = @SVector([zero(T), T(-18.75e3 / 1.25e-2)])
    
    "Photoionization model."
    phmodel = empty_photoionization(T)
    
    "Transport model."
    trans = minimal_transport_model(T)
    
    "Chemical model."
    chem = NetIonization(minimal_transport_model(T))
    
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
    
    "Stop refinement beyond a certain r."
    refine_max_r = nothing
    
    "The type of flux scheme (:koren / :weno)"
    flux_scheme::Symbol = :koren

    "Interp. of density ghosts at ref. boundaries (:limited/ :linear/ :pseudoquadratic/ :copy)"
    dens_ghost_interp::Symbol = :limited

    "Interp. of potential at ghosts at ref. boundaries (:limited/ :linear/ :pseudoquadratic/ :copy)"
    poisson_ghost_interp::Symbol = :pseudoquadratic

    "Interp. of charge-density ghosts at ref. boundaries (default `dens_ghost_interp`)"
    charge_ghost_interp::Union{Symbol, Nothing} = nothing

    """
    Interp. of densities into newly refined blocks (:positive / :limited / :linear /
    :pseudoquadratic / :copy).  Unlike the ghost interpolations this one becomes part of the
    state, so a slope-flattening method (:limited, :copy) acts as a numerical diffusion that
    blunts the streamer head every time the refinement cascade follows it.
    """
    dens_refine_interp::Symbol = :positive

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

    "Time stepper (:euler / :midpoint / :heun / :ssprk3 / :ssprk43)."
    time_stepper::Symbol = :ssprk3
    
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
    
    "Input file"
    _input::String = ""

    "Files included from the input files (copied to the output folder)."
    _included::Vector{String} = String[]

    "Name for the simulation."
    name::String = replace(basename(_input), r"(\.in)?\.jl$" => "")

    "An identifier for this simulation."
    _uid::String = String(rand('A':'Z', 3)) * "-" * Dates.format(now(), "yyyymmdd-HHMMSS")

    "Where to write output data"
    outfolder = joinpath(splitdir(_input)[1], name, _uid)

    "Functions to call at output snapshots."
    output_callbacks=()

    "Function `(t, conf, fields, tree, conn) -> Bool` checked at output times; run stops if true."
    stop_condition = nothing

    "Whether this is a debug run"
    debug = false
    
    "Start date of run"
    _date = now()

    "Git commit"
    _git_commit = "UNKNOWN"

    "Is git repo dirty?"
    _git_dirty = true
end


"""
Main entry point when run as a script. ARGS must contain a list with input files (read and updated
in order). The function also admits extra keyword parameters in kw... that are useful for debug.

The parameters a received in `input` and possibly updated with additional keyword arguments.
Generally `input` should be of type `InputParameters` but you can use any type woth the appropriate
fields.
"""
function (@main)(ARGS; kw...)
    s = ArgParseSettings()
        
    @add_arg_table! s begin
        "input_file"
        help = "Input file with parameters"
        required = true
        nargs = '+'
    end
        
    args = parse_args(ARGS, s)
    
    allowed = [:T, allowedinps(_main)..., allowedinps(InputParameters)...]
    input = Dict{Symbol, Any}(:T => Float64)
    included = String[]
    for input_file in args["input_file"]
        inp = read_input(input_file, allowed)
        append!(included, get(inp, :_included, String[]))
        merge!(input, inp)
    end
    input[:_included] = unique(included)
    
    merge!(input, pairs(kw))

    # Convert to InputParameters to enforce types and defaults
    local params1
    try
        # Must remove T because it's not allowed as kw
        T = pop!(input, :T)
        params1 = InputParameters{T}(;input...)
    catch e
        @error "Error in the input parameters (see backtrace)"
        rethrow(e)
    end

    io = IOBuffer()
    pprint(IOContext(io, :color => true), params1)
    @info "Input parameters" * "\n" * String(take!(io))

    local ret
    with_logger(TerminalLogger(meta_formatter=metafmt)) do
        # invokelatest allows functions newer than this to be called within the simulation.
        ret = Base.invokelatest(_main, params1)
    end

    if params1.debug
        return ret
    else
        return 0
    end
end


function _main(input::InputParameters{T}) where T
    (;D, M, L) = input

    if input._git_dirty
        @warn """
              Your git repo is dirty (you have uncommitted changes).
              To make runs reproducible it is strongly recommended that you run in a clean repo
              so that the commit id can be stored.
              """
        sleep(5)
    end

    Polyester.reset_threads!()

    # Set flux scheme Koren / WENO
    fluxschem = (input.flux_scheme == :koren ? FluxSchemeKoren() :
        input.flux_scheme == :weno ? FluxSchemeWENO() :
        throw(ArgumentError("Unknown flux scheme $(input.flux_scheme)")))
    
    # Koren needs 2 ghosts; WENO needs 4
    G = needghost(fluxschem)

    # Interpolation methods
    dens_ghost_interp = getinterp(input.dens_ghost_interp, true)
    dens_refine_interp = getinterp(input.dens_refine_interp, true)
    poisson_ghost_interp = getinterp(input.poisson_ghost_interp)
    charge_ghost_interp = getinterp(something(input.charge_ghost_interp, input.dens_ghost_interp))

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
    geom = (D == 2 ? CylindricalGeometry{1, T}(input.rootsize[1] * M * (1 << (refine_maxlevel - 1))) :
            CartesianGeometry())
    lpl = LaplacianDiscretization{D, input.poisson_order}()

    stencil = (input.poisson_order == 2 ? StarStencil{D}() :
        input.poisson_order == 4 ? BoxStencil{D}() :
        throw(ArgumentError("poisson_order = $(input.poisson_order) not allowed")))
        
    # Float64 model parameters (e.g. BagheriTransportModel()) would promote the kernels.
    trans = _convert_model(T, input.trans)
    chem = _convert_model(T, input.chem)

    # The data struct that contains all streamer fields
    stepper = timestepper(input.time_stepper)
    fields = StreamerFields(T, nspecies(chem), length(input.phmodel), D, M, G, Val(input.storage);
                            stepper)
    freebcinst = input.freebnd ? FreeBC(geom, fields.u, refine_maxlevel, input.rootsize, h) : nothing
    
    ###
    #  REFINEMENT CRITERIUM
    ###
    if isnothing(input.refinement)
        # The Teunissen refinement criterium based on the electric field
        tref = TeunissenRef(fields.eabs, trans,
                            input.refine_teunissen_c0,
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

        if !isnothing(input.refine_max_r)
            ref = OrRef(DirThresholdRef{1, M, T}(input.refine_max_r, input.refine_min_h), ref)
        end
    else
        ref = input.refinement
        nref = input.refinement
    end

    # Allow users to give eb as Vector. Performance impact sould be negligible as everything is done
    # below function boundaries.
    if input.eb isa AbstractVector
        eb = SVector{length(input.eb), T}(input.eb)
    else
        eb = input.eb
    end
    
    # Set the streamer configuration
    conf = StreamerConf(h, eb, geom, fbc, pbc, lpl, freebcinst, trans, fluxschem,
                        dens_ghost_interp, dens_refine_interp, poisson_ghost_interp,
                        charge_ghost_interp, chem, input.phmodel, input.dens, ref, stencil,
                        input.dt_safety_factor, input.poisson_fmg, input.poisson_iter,
                        input.poisson_level_iter)
    
    # Start with a full tree up to level 2
    populate!(tree, input.init_populate_level)
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

        isnew = isempty(at)
        push!(az, zemax)
        push!(at, t)
        push!(aemax, emax)

        # Appended at every output time so that the history survives an interrupted run.
        open(joinpath(input.outfolder, "emax.csv"), isnew ? "w" : "a") do io
            isnew && writedlm(io, ["t" "zemax" "emax"], ',')
            writedlm(io, [t zemax emax], ',')
        end
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
    
    if !input.debug
        if !isdir(input.outfolder)
            mkpath(input.outfolder)
            @info "$(input.outfolder) created"
        end

        writejl(joinpath(input.outfolder, input.name * ".ref.jl"), input)
        cp(input._input, joinpath(input.outfolder, input.name * ".in.jl"))
        _archive_included(input._included, input.outfolder)
    
        run!(fields, conf, tree, conn, stepper, input.tend; progress_every=10, input.output,
             derefine_minlevel,
             refine_maxlevel,
             output_callbacks, onerror, input.stop_condition)
    end

    return NamedTuple(Base.@locals)
end


"""
Copy the files in `included` into `outfolder/included`, together with a list of their original
paths. Files with repeated names are prefixed with their position in the list.
"""
function _archive_included(included, outfolder)
    isempty(included) && return
    dir = joinpath(outfolder, "included")
    mkpath(dir)
    names = basename.(included)
    for (i, (path, name)) in enumerate(zip(included, names))
        target = count(==(name), names) > 1 ? "$(i)-$(name)" : name
        cp(path, joinpath(dir, target); force=true)
    end
    write(joinpath(dir, "paths.txt"), join(included, "\n") * "\n")
end
