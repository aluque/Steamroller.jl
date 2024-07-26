#=
Code to produce a streamer simulation.
=#

"""
Fields required to run a streamer fluid simulation with K species (electrons are assumed to be first).

* `T` is the floating-point type (e.g. Float64 or Float32).
* `K` is the number of species.
* `L` is the number of photoionization fields (can be zero).
* `S` is the type of the scalar fields.
* `V` is the type of the vector fields.
* `M` is the type of the scalar field for the refinement marker.
* `R` is the type of the scalar field for the change in refinement level (different type bc 
    it contains a single scalar per block.)
"""
struct StreamerFields{T, K, L, S <: ScalarBlockField,
                      V <: VectorBlockField, M <: ScalarBlockField,
                      R <: ScalarBlockField}
    "Species density"
    n::SVector{K, S}

    "Intermediate species density"
    n1::SVector{K, S}
    
    "Derivatives of the densities (several of them b.c. we use RK)"
    dn::Vector{SVector{K, S}}
        
    "Charge density"
    q::S

    "Helper charge density (only for higher order poisson solvers)"
    q1::S
    
    "Electrostatic Potential"
    u::S

    "Helper electrostatic potential field"
    u1::S

    "Residual in the poisson equation"
    r::S
    
    "Electric field magnitude"
    eabs::S

    "Electric field"
    e::V

    "Flux vector.  In many cases can be the same as the electric field vector"
    flux::V

    "Photoionization fields."
    photo::SVector{L, S}
    
    "Refinement marker"
    m::M

    "Refinement delta"
    refdelta::R

    flux_same_as_e::Bool

    maxdt::Vector{T}

    "Available blocks"
    freeblocks::Vector{Int}
    
    """
    Initialize a `StreamerFields` instance.
    """
    function StreamerFields(T, K, L, D, M, G, storage=Val(:contiguous), flux_same_as_e=false)
        kscalars(k) = SVector{K}(ntuple(i -> ScalarBlockField{D, M, G, T}(storage), Val(K)))
        n = kscalars(K)
        n1 = kscalars(K)
        dn = [kscalars(K) for _ in 1:3]
        
        q = ScalarBlockField{D, M, G, T}(storage)
        q1 = ScalarBlockField{D, M, G, T}(storage)
        
        u = ScalarBlockField{D, M, G, T}(storage)
        u1 = ScalarBlockField{D, M, G, T}(storage)
        r = ScalarBlockField{D, M, G, T}(storage)

        e = VectorBlockField{D, M, G, T}(storage)

        flux = flux_same_as_e ? e : VectorBlockField{D, M, G, T}(storage)

        photo = SVector{L}(ntuple(i -> ScalarBlockField{D, M, G, T}(storage), Val(L)))
        
        eabs = ScalarBlockField{D, M, G, T}(storage)
        m = ScalarBlockField{D, M, 3, T}(storage)
        refdelta = ScalarBlockField{D, 1, 0, RefDelta}(Val(:contiguous))

        maxdt = Vector{T}(undef, 2^14)
        freeblocks = Int[]
        
        return new{
            T, K, L, typeof(q), typeof(e), typeof(m), typeof(refdelta)
        }(n, n1, dn, q, q1, u, u1, r, eabs, e, flux, photo, m, refdelta,
          flux_same_as_e, maxdt, freeblocks)
    end
end

nspecies(::StreamerFields{T, K}) where {T, K} = K

"""
Build an `SVector` with all species at a given grid point.
"""
function species(n::SVector{K}, I, blk) where {K}
    SVector{K}(ntuple(i -> n[i][I, blk], Val(K)))
end


"""
Add a block to each field in the `StreamerFields` set of fields.
"""
function newblock!(sf::StreamerFields, j0=nothing)
    (;n, n1, dn, q, q1, u, u1, r, eabs, e, flux, photo, m, refdelta, flux_same_as_e) = sf
    
    # We obtain the block number of a field and then check that all fields create the same block number
    j = newblock!(q)
    !isnothing(j0) && @assert(j0 == j)
    
    foreach(i->newblock!(n[i], j), eachindex(n))
    foreach(i->newblock!(n1[i], j), eachindex(n1))
    for dni in dn
        foreach(i->newblock!(dni[i], j), eachindex(dni))
    end
        
    newblock!(q1, j)
    newblock!(u, j)
    newblock!(u1, j)
    newblock!(r, j)
    newblock!(eabs, j)
    newblock!(e, j)
    flux_same_as_e || newblock!(flux, j)
    foreach(i->newblock!(photo[i], j), eachindex(photo))

    newblock!(m, j)
    newblock!(refdelta, j)

    m[j] .= 0
    
    # When debugging it is useful to initialize everything as NaN to make sure that the initialization
    # values are never used.
    if false
        for f in [q, q1, u, u1, r, eabs, e, flux]
            f[j] .= NaN       
        end
        
        for i in eachindex(n)
            n[i][j] .= NaN
            n1[i][j] .= NaN
        end
        
        for i in eachindex(photo)
            photo[i][j] .= NaN
        end
        
        for dni in dn
            for dnii in dni
                dnii[j] .= NaN
            end
        end
    end
    
    return j
end


# We enable calls to newblocks using tuples. This allows us to pass to refine! something besides
# the StreamerField.
newblock!(::Tuple{}, j) = j
newblock!(t::Tuple, j) = (newblock!(t[1], j); newblock!(Base.tail(t), j))
newblock!(::Nothing, j=nothing) = j
newblock!(t::Tuple) = (j = newblock!(t[1]); newblock!(Base.tail(t), j))


function newblocks!(f, nblocks)
    for i in 1:nblocks
        newblock!(f)
    end
end


"""
Interpolate all fields that require interpolation in `sf`, from block `src` into `dest`, with
`sub` being the sub-block local coordinates.
"""
function interp!(sf::StreamerFields, dest, src, sub)
    (;n, u, photo, m) = sf
    for ni in n
        interp!(ni[dest], validindices(ni), ni[src], subblockindices(ni, sub))
    end
    
    interp!(u[dest], validindices(u), u[src], subblockindices(u, sub))

    for phf in photo
        interp!(phf[dest], validindices(phf), phf[src], subblockindices(phf, sub))
    end
    
    # Refinement marking are initialized as zero
    m[dest] .= 0
end

interp!(::Nothing, dest, src, sub) = nothing
interp!(::Tuple{}, dest, src, sub) = nothing
interp!(t::Tuple, dest, src, sub) = (interp!(t[1], dest, src, sub); interp!(Base.tail(t), dest, src, sub))


"""
Restrict all fields that require restriction in `sf`, from block `src` into `dest`, with
`sub` being the sub-block local coordinates.
"""
function restrict!(sf::StreamerFields, dest, src, sub)
    (;n) = sf
    for ni in n
        restrict!(ni[dest], subblockindices(ni, sub), ni[src], validindices(ni))
    end
end

restrict!(::Nothing, dest, src, sub) = nothing
restrict!(::Tuple{}, dest, src, sub) = nothing
restrict!(t::Tuple, dest, src, sub) = (restrict!(t[1], dest, src, sub); restrict!(Base.tail(t), dest, src, sub))

"""
Configuration of a streamer simulation
"""
struct StreamerConf{T,
                    EB,
                    G  <: AbstractGeometry,
                    BC,
                    L  <: LaplacianDiscretization,
                    FBC, # Free b.c.
                    TR <: AbstractTransportModel,
                    F  <: AbstractFluxScheme,
                    CH <: AbstractChemistry,
                    PH <: PhotoionizationModel,
                    S  <: AbstractDensityScaling,
                    R  <: AbstractRefinement,
                    ST}
    "Spatial discretization at level 1"
    h::T

    "Background electric field"
    eb::EB
    
    "Geometry"
    geom::G
    
    "Boundary conditions for flux"
    fbc::BC

    "Boundary conditions for Poisson"
    pbc::BC
    
    "Laplacian discretiation"
    lpl::L

    "Free boundary conditions (or nothing if not applicable)"
    freebnd::FBC
    
    "Transport model"
    trans::TR

    "Flux discretization scheme"
    fluxschem::F
    
    "Chemistry model"
    chem::CH

    "Photoionization model"
    phmodel::PH
    
    "Density scaling"
    dens::S
    
    "Refinement criterium"
    ref::R

    "Stencil to compute connectivities"
    stencil::ST
    
    "Safety factor below dt limits"
    dt_safety_factor::T

    "Use full multigrid for the Poisson equation?"
    poisson_fmg::Bool

    "Number of Poisson iterations"
    poisson_iter::Int

    "(up, down, top) iterations in V cycles in Poisson."
    poisson_level_iter::NTuple{3, Int}
end


"""
Compute derivatives starting from densities ni and stores them in dni.
"""
function derivs!(dni, ni, t, fld::StreamerFields, conf::StreamerConf{T},
                 tree, conn) where {T}
    (;q, q1, r, q, u, u1, e, flux, photo, eabs, maxdt) = fld
    (;eb, h, trans, fluxschem, chem, phmodel, dens, geom, lpl, freebnd, pbc, fbc) = conf
    (;poisson_fmg, poisson_iter, poisson_level_iter) = conf
    (nup, ndown, ntop) = poisson_level_iter
    
    netcharge!(tree, q, ni, chem)

    # This is only needed for higher-order discretization so one can save a bit of time
    # by not doing this restriction always.
    restrict_full!(q, conn)
    for l in 1:length(tree)
        fill_ghost!(q, l, conn, pbc)
        rhs_level!(q1, q, tree[l], lpl, geom)
        fill_ghost!(q1, l, conn, pbc)
    end

    setfreebc!(freebnd, pbc, fld, conf, tree, conn)
    
    for i in 1:poisson_iter
        for l in 1:length(tree)
            fill_ghost!(u, l, conn, pbc)
        end

        if poisson_fmg
            fmg!(u, q1, r, u1, h^2 * convert(T, co.elementary_charge / co.epsilon_0),
                 tree, conn, geom, pbc, lpl; nup, ndown, ntop)
        else
            vcycle!(u, q1, r, u1, h^2 * convert(T, co.elementary_charge / co.epsilon_0),
                    tree, conn, geom, pbc, lpl; nup, ndown, ntop)
        end
    end
    
    # The electron density is always first among species.
    ne = ni[1]
    restrict_full!(ne, conn)

    for l in 1:length(tree)
        fill_ghost_copy!(ne, conn.neighbor[l])
        fill_ghost_bnd!(ne, conn.boundary[l], fbc)
        fill_ghost_interp!(ne, conn.refboundary[l], InterpCopy())

        fill_ghost_copy!(u, conn.neighbor[l])
        fill_ghost_bnd!(u, conn.boundary[l], pbc)
        fill_ghost_interp!(u, conn.refboundary[l])
        fill_ghost_free!(u, freebnd, tree, l)
    end
    
    electric_field!(tree, e, eabs, u, h, t, eb)
    
    for l in 1:length(tree)
        fill_ghost!(eabs, l, conn, ExtrapolateConst())
    end
    
    resize!(maxdt, max(length(maxdt), length(ne)))
    flux!(tree, fluxschem, flux, ne, e, eabs, h, trans, dens, maxdt)
    restrict_flux!(flux, conn)

    chemderivs!(tree, dni, ni, eabs, h, chem, dens, Val{:pre}(), Val{true}())    
    # Here goes photo-ionization
    photoionization!(tree, dni, photo, r, u1, phmodel, h, conn, geom)
    chemderivs!(tree, dni, ni, eabs, h, chem, dens, Val{:post}(), Val{false}())
    fluxderivs!(tree, dni[1], flux, h, geom)
end


@bkernel function netcharge!((tree, level, blkpos, blk), q, n, chem)
    for I in validindices(n[1])
        q[I, blk] = netcharge(chem, species(n, I, blk))
    end
end


"""
Perform a single step of the time integration.

# Arguments
- `fld::StreamerFields`: the fields to be updated.
- `conf::StreamerConf`: the streamer configuration.
- `tree`: a `Tree` instance containing the tree.
- `conn`: connectivity pattern of the tree.
- `dt`: time step.
"""
function step!(fld::StreamerFields{T, K}, conf::StreamerConf{T}, tree, conn,
               t, tmax, ::Val{:ssprk3}) where {T, K}
    (;dn, n, n1, maxdt) = fld
    (;dt_safety_factor, fbc) = conf

    maxdt .= typemax(eltype(maxdt))
    
    ## This is SSPRK (3rd order SSP Runge-Kutta).
    derivs!(dn[1], n, t, fld, conf, tree, conn)

    # Compute the dt
    dt = min(tmax - t, dt_safety_factor * mapreduce_tree((lvl, I, blk) -> maxdt[blk], min, tree, typemax(T)))
    for s in 1:K
        n1[s] .= n[s] .+ dn[1][s] .* dt
    end
        
    derivs!(dn[2], n1, t + dt, fld, conf, tree, conn)
    dt1 = dt / 4

    for s in 1:K
        # Here and below there's some bug in VectorOfArray that adds a lot of overhead in broadcasting
        # if I remove the brackets.
        n1[s] .= n[s] .+ (dn[1][s] .* dt1 .+ dn[2][s] .* dt1)
    end

    derivs!(dn[3], n1, t + dt / 2, fld, conf, tree, conn)

    for s in 1:K
        n[s] .= n[s] .+ (dn[1][s] .* (dt / 6) .+ dn[2][s] .* (dt / 6) .+ dn[3][s] .* (2 * dt / 3))
    end

    for s in 1:K
        restrict_full!(n[s], conn)
        for l in 1:length(tree)
            fill_ghost!(n[s], l, conn, fbc)
        end
    end
        
    return (t + dt, dt)
end


function refine!(fields::StreamerFields, conf::StreamerConf{T}, tree, conn, t, dt;
                 minlevel=1, maxlevel=typemax(Int), ref=nothing) where {T}
    (;m, refdelta, n, freeblocks) = fields
    (;h, fbc, freebnd) = conf

    ref = isnothing(ref) ? conf.ref : ref

    refmark!(tree, m, ref, h, t, dt)
    fill_ghost_copy!(m, conn)
    refdelta!(tree, refdelta, m, BoxStencil{dimension(m)}())

    for ni in n
        fill_ghost_copy!(ni, conn)
        fill_ghost_bnd!(ni, conn, fbc)
        fill_ghost_interp!(ni, conn)
    end
        
    applydelta!(tree, refdelta, freeblocks, (fields, freebnd), minlevel, maxlevel)
end

"""
Sets the initial conditions and keeps refining according to criterium `ref` until no new blocks
are created.  The initial conditions are specified as a vector of pairs 
`species => func` where `species` is a species number and `func` is a function from `D` variables to
a scalar, where `D` is the dimensionality.  Remaining `kwargs` are passed to refine!
(e.g. minlevel, maxlevel).

Returns the new connectivity.
"""
function initial_conditions!(fields::StreamerFields{T, K}, conf, tree, ref, conditions; kwargs...) where {T, K}
    (;h, stencil, fbc) = conf
    (;freeblocks, n, photo) = fields

    
    prevn = 0
    conn = connectivity(tree, stencil)
    while prevn != nblocks(tree)
        prevn = nblocks(tree)

        for (species, func) in conditions
            maptree!(func, n[species], tree, h)
        end

        for i in 1:K
            if count(==(i), first.(conditions)) == 0
                maptree!(Background{T}(0.0), n[i], tree, h)
            end
        end
        
        refine!(fields, conf, tree, conn, T(0.0), T(Inf); ref, kwargs...)        
        connectivity!(conn, tree, stencil)
    end

    for ni in n
        fill_ghost_copy!(ni, conn)
        fill_ghost_bnd!(ni, conn, fbc)
        fill_ghost_interp!(ni, conn)
    end

    for ph in photo
        ph .= 0
    end
    
    fields.flux .= 0
    fields.u .=0
    fields.u1 .= 0
    fields.r .= 0
    fields.eabs .= 0
    fields.q .= 0
    fields.q1 .= 0
    fields.e .= 0
    
    return conn
end


"""
Execute the full streamer simulation.
"""
function run!(fields::StreamerFields{T}, conf, tree, conn, tend;
              min_dt=1e-16, output=[],
              derefine_minlevel=3, refine_maxlevel=10,
              refine_every=2, progress_every=50,
              output_callbacks=(),
              onerror=()) where {T}
    (;stencil, h) = conf
    output = map(x->convert(T, x), output)

    # Measure times
    elapsed_step = 0.0
    elapsed_refine = 0.0
    elapsed_connectivity = 0.0
    
    t = zero(T)
    dt = zero(T)
    local msg
    iter = 0
    start = time()

    
    @withprogress begin
        try
            while t < tend
                tfinal = get(output, 1, convert(T, Inf))
                
                elapsed_step += @elapsed (t, dt) = step!(fields, conf, tree, conn, t, tfinal,
                                                         Val(:ssprk3))
                if t > 0 && dt < min_dt
                    @error "dt is below the minimal allowed min_dt.  Stopping the iterations here." dt min_dt
                    throw(ErrorException("dt below min_dt"))
                end
                
                if !isempty(output) && isapprox(t, first(output))
                    _run_callbacks(output_callbacks, t, conf, fields, tree, conn)
                    popfirst!(output)
                end
                
                if (iter % refine_every) == 0
                    elapsed_refine += @elapsed refine!(fields, conf, tree, conn, t, dt;
                                                       minlevel=derefine_minlevel, maxlevel=refine_maxlevel)
                    elapsed_connectivity += @elapsed connectivity!(conn, tree, stencil)
                end
                
                
                if (iter % progress_every) == 0
                    @logprogress iter * dt / (iter * dt + tend - t) #(t / tend)
                    msg = _msg(;h, t, dt, iter, tree, elapsed_step, elapsed_refine,
                               elapsed_connectivity, start)
                    #percent = format("{:.1f}", 100 * t / tend)
                    @info "\n```\n$msg\n```" sticky=true _id=:report
                end
                
                iter += 1
            end
        catch e
            empty!(Logging.current_logger().sticky_messages)
            msg = _msg(;h, t, dt, iter, tree, elapsed_step, elapsed_refine, elapsed_connectivity, start)
            @error "An error occurred within the simulation (see stacktrace)\n```\n$msg\n```"
            if !isempty(onerror)
                _run_callbacks(onerror, t, conf, fields, tree, conn)
            end
            rethrow(e)
        end
    end

    empty!(Logging.current_logger().sticky_messages)

    msg = _msg(;h, t, dt, iter, tree, elapsed_step, elapsed_refine, elapsed_connectivity, start)
    @info "Simulation completed\n```\n$msg\n```"

    return (;t, dt, iter, elapsed_step, elapsed_refine, elapsed_connectivity)
end


function _msg(;h, t, dt, iter, tree, elapsed_step, elapsed_refine, elapsed_connectivity, start)
    join(map(x -> @sprintf("%30s = %-30s", string(first(x)), repr(last(x))), 
             Pair{Symbol, Any}[:t => t,
                               :dt => dt,
                               :iter => iter,
                               :max_level => findlast(!isempty, tree),
                               :min_h => h / 2^(findlast(!isempty, tree) - 1),
                               :nblocks => nblocks(tree),
                               :running_time => time() - start,
                               :elapsed_step => elapsed_step,
                               :elapsed_refine => elapsed_refine,
                               :elapsed_connectivity => elapsed_connectivity]), "\n")
end


_run_callbacks(f::Tuple{}, args...) = nothing
function _run_callbacks(f::Tuple, args...)
    first(f)(args...)
    _run_callbacks(Base.tail(f), args...)
end

