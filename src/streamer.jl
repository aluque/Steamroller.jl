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
    
    """
    Initialize a `StreamerFields` instance.
    """
    function StreamerFields(T, K, L, D, M, H, storage=Val(:contiguous), flux_same_as_e=false)
        TH = SVector{H, T}

        kscalars(k) = SVector{K}(ntuple(i -> ScalarBlockField{D, M, 2, T}(storage), Val(K)))
        n = kscalars(K)
        n1 = kscalars(K)
        dn = [kscalars(K) for _ in 1:3]
        
        q = ScalarBlockField{D, M, 2, T}(storage)
        q1 = ScalarBlockField{D, M, 2, T}(storage)
        
        u = ScalarBlockField{D, M, 2, T}(storage)
        u1 = ScalarBlockField{D, M, 2, T}(storage)
        r = ScalarBlockField{D, M, 2, T}(storage)

        e = VectorBlockField{D, M, 2, T}(storage)

        flux = flux_same_as_e ? e : VectorBlockField{D, M, 2, T}(storage)

        photo = SVector{L}(ntuple(i -> ScalarBlockField{D, M, 2, T}(storage), Val(L)))
        
        eabs = ScalarBlockField{D, M, 2, T}(storage)
        m = ScalarBlockField{D, M, 3, T}(storage)
        refdelta = ScalarBlockField{D, 1, 0, RefDelta}(Val(:contiguous))

        maxdt = Vector{T}(undef, 2^14)
        
        return new{
            T, K, L, typeof(q), typeof(e), typeof(m), typeof(refdelta)
        }(n, n1, dn, q, q1, u, u1, r, eabs, e, flux, photo, m, refdelta, flux_same_as_e, maxdt)
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
function newblock!(sf::StreamerFields)
    (;n, n1, dn, q, q1, u, u1, r, eabs, e, flux, photo, m, refdelta, flux_same_as_e) = sf
    
    # We obtain the block number of a field and then check that all fields create the same block number
    j = newblock!(q)

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

    return j
end

function newblocks!(sf::StreamerFields, nblocks)
    for i in 1:nblocks
        newblock!(sf)
    end
end


"""
Interpolate all fields that require interpolation in `sf`, from block `src` into `dest`, with
`sub` being the sub-block local coordinates.
"""
function interp!(sf::StreamerFields, dest, src, sub)
    (;n, u, m) = sf
    for ni in n
        interp!(ni[dest], validindices(ni), ni[src], subblockindices(ni, sub))
    end
    
    interp!(u[dest], validindices(u), u[src], subblockindices(u, sub))

    # Refinement marking are initialized as zero
    m[dest] .= 0
end


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


"""
Configuration of a streamer simulation
"""
struct StreamerConf{T, D,
                    G  <: AbstractGeometry,
                    BC,
                    L  <: LaplacianDiscretization,
                    TR <: AbstractTransportModel,
                    CH <: AbstractChemistry,
                    PH <: PhotoionizationModel,
                    R  <: AbstractRefinement}
    "Spatial discretization at level 1"
    h::T

    "Background electric field"
    eb::SVector{D, T}
    
    "Geometry"
    geom::G
    
    "Boundary conditions for flux"
    fbc::BC

    "Boundary conditions for Poisson"
    pbc::BC
    
    "Laplacian discretiation"
    lpl::L

    "Transport model"
    trans::TR

    "Chemistry model"
    chem::CH

    "Photoionization model"
    phmodel::PH
    
    "Refinement criterium"
    ref::R

    "Safety factor below dt limits"
    dt_safety_factor::T
end


"""
Compute derivatives starting from densities ni and stores them in dni.
"""
function derivs!(dni, ni, t, fld::StreamerFields, conf::StreamerConf{T},
                 tree, conn, fmgiter=2) where {T}
    (;q, q1, r, q, u, u1, e, flux, photo, eabs, maxdt) = fld
    (;eb, pbc, h, trans, chem, phmodel, geom, lpl, pbc, fbc) = conf
    
    netcharge!(tree, q, ni, chem)

    for l in 1:length(tree)
        fill_ghost!(q, l, conn, pbc)
        rhs_level!(q1, q, tree[l], lpl, geom)
        fill_ghost!(q1, l, conn, pbc)
    end

    for i in 1:fmgiter
        for l in 1:length(tree)
            fill_ghost!(u, l, conn, pbc)
        end

        vcycle!(u, q1, r, u1, h^2 * convert(T, co.elementary_charge / co.epsilon_0),
                tree, conn, geom, pbc, lpl, nup=2, ndown=2, ntop=4)
        # fmg!(u, q1, r, u1, h^2 * convert(T, co.elementary_charge / co.epsilon_0),
        #      tree, conn, geom, pbc, lpl, nup=3, ndown=3, ntop=1)
    end

    # The electron density is always first among species.
    ne = ni[1]
    for l in 1:length(tree)        
        fill_ghost_copy!(ne, conn.neighbor[l])
        fill_ghost_bnd!(ne, conn.boundary[l], fbc)
        fill_ghost_interp!(ne, conn.refboundary[l], InterpCopy())

        fill_ghost!(u, l, conn, pbc)
    end
    
    electric_field!(tree, e, eabs, u, h, eb)

    for l in 1:length(tree)
        fill_ghost!(eabs, l, conn, ExtrapolateConst())
    end

    resize!(maxdt, max(length(maxdt), length(ne)))
    flux!(tree, flux, ne, e, eabs, h, trans, maxdt)
    restrict_flux!(flux, conn)

    chemderivs!(tree, dni, ni, eabs, chem, Val{true}(), Val{true}())    
    # Here goes photo-ionization
    photoionization!(tree, dni, photo, r, u1, phmodel, h, conn, geom)
    chemderivs!(tree, dni, ni, eabs, chem, Val{false}(), Val{false}())
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
- `fmgiter=1`: number of FMG iterations.
"""
function step!(fld::StreamerFields{T, K}, conf::StreamerConf{T}, tree, conn,
               t, tmax, ::Val{:ssprk3}) where {T, K}
    (;dn, n, n1, maxdt) = fld
    (;dt_safety_factor) = conf

    maxdt .= typemax(eltype(maxdt))
    
    ## This is SSPRK (3rd order SSP Runge-Kutta).
    derivs!(dn[1], n, t, fld, conf, tree, conn)

    # Compute the dt
    dt = min(tmax - t, dt_safety_factor * mapreduce((lvl, I, blk) -> maxdt[blk], min, tree, typemax(T)))
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
    end

    return (t + dt, dt)
end


function refine!(fld::StreamerFields, conf::StreamerConf{T, D}, tree, conn, t, dt, freeblocks;
                 minlevel=1, maxlevel=typemax(Int), ref=nothing) where {T, D}
    (;m, refdelta, n) = fld
    (;h, fbc) = conf

    ref = isnothing(ref) ? conf.ref : ref

    refmark!(tree, m, ref, h, t, dt)
    fill_ghost_copy!(m, conn)
    refdelta!(tree, refdelta, m, BoxStencil{D}())

    for ni in n
        fill_ghost_copy!(ni, conn)
        fill_ghost_bnd!(ni, conn, fbc)
        fill_ghost_interp!(ni, conn)
    end
        
    applydelta!(tree, refdelta, freeblocks, fld, minlevel, maxlevel)
end

