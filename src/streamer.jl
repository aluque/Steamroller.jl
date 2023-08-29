#=
Code to produce a streamer simulation.
=#

"""
Fields required to run a streamer fluid simulation.
"""
struct StreamerFields{T, S <: ScalarBlockField, SH <: ScalarBlockField,
                      V <: VectorBlockField, M <: ScalarBlockField,
                      R <: ScalarBlockField}
    "Electron density"
    ne::S

    # Note that nh are of SCALAR type even if they store several numbers per cell.
    # here SCALAR is used in the physicist's sense.
    "Heavy species density"
    nh::SH

    "Electron density for intermediate eval"
    ne1::S

    "Heavy species density for intermediate eval"
    nh1::SH

    "Derivatives of the electron density and the heavy density"    
    ke::SVector{3, S}
    kh::SVector{3, SH}
        
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
    
    "Refinement marker"
    m::M

    "Refinement delta"
    refdelta::R

    flux_same_as_e::Bool

    maxdt::Vector{T}
    
    """
    Initialize a `StreamerFields` instance.
    """
    function StreamerFields(T, D, M, H, flux_same_as_e=false)
        TH = SVector{H, T}
        ne = ScalarBlockField{D, M, 2, T}()
        ne1 = ScalarBlockField{D, M, 2, T}()
        nh = ScalarBlockField{D, M, 2, TH}()
        nh1 = ScalarBlockField{D, M, 2, TH}()
        
        ke = @SVector [ScalarBlockField{D, M, 2, T}(),
                       ScalarBlockField{D, M, 2, T}(),
                       ScalarBlockField{D, M, 2, T}()]
        
        kh = @SVector [ScalarBlockField{D, M, 2, TH}(),
                       ScalarBlockField{D, M, 2, TH}(),
                       ScalarBlockField{D, M, 2, TH}()]
                       
        q = ScalarBlockField{D, M, 2, T}()
        q1 = ScalarBlockField{D, M, 2, T}()
        
        u = ScalarBlockField{D, M, 2, T}()
        u1 = ScalarBlockField{D, M, 2, T}()
        r = ScalarBlockField{D, M, 2, T}()

        e = VectorBlockField{D, M, 2, T}()

        flux = flux_same_as_e ? e : VectorBlockField{D, M, 2, T}()

        eabs = ScalarBlockField{D, M, 2, T}()
        m = ScalarBlockField{D, M, 3, Bool}()
        refdelta = ScalarBlockField{2, 1, 0, RefDelta}()        

        maxdt = Vector{T}(undef, 2^14)
        
        return new{
            T, typeof(ne), typeof(nh), typeof(e), typeof(m), typeof(refdelta)
        }(ne, nh, ne1, nh1, ke, kh, q, q1, u, u1, r, eabs, e, flux, m, refdelta, flux_same_as_e, maxdt)
    end
end


"""
Add a block to each field in the `StreamerFields` set of fields.
"""
function newblock!(sf::StreamerFields)
    (;ne, nh, ne1, nh1, ke, kh, q, q1, u, u1, r, eabs, e, flux, m, refdelta, flux_same_as_e) = sf
    
    n = newblock!(ne)
    newblock!(nh, n)
    newblock!(ne1, n)
    newblock!(nh1, n)
    foreach(i->newblock!(ke[i], n), eachindex(ke))
    foreach(i->newblock!(kh[i], n), eachindex(kh))
    
    newblock!(q, n)
    newblock!(q1, n)
    newblock!(u, n)
    newblock!(u1, n)
    newblock!(r, n)
    newblock!(eabs, n)
    newblock!(e, n)
    flux_same_as_e || newblock!(flux, n)
    newblock!(m, n)
    newblock!(refdelta, n)

    return n
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
    (;ne, nh, u) = sf
    
    interp!(ne[dest], validindices(ne), ne[src], subblockindices(ne, sub))
    interp!(nh[dest], validindices(nh), nh[src], subblockindices(nh, sub))
    interp!(u[dest], validindices(u), u[src], subblockindices(u, sub))
end


"""
Restrict all fields that require restriction in `sf`, from block `src` into `dest`, with
`sub` being the sub-block local coordinates.
"""
function restrict!(sf::StreamerFields, dest, src, sub)
    (;ne, nh) = sf
    
    restrict!(ne[dest], subblockindices(ne, sub), ne[src], validindices(ne))
    restrict!(nh[dest], subblockindices(nh, sub), nh[src], validindices(nh))
end


"""
Configuration of a streamer siumation
"""
struct StreamerConf{T, D,
                    G  <: AbstractGeometry,
                    BC,
                    L  <: LaplacianDiscretization,
                    TR <: AbstractTransportModel,
                    CH <: AbstractChemistry,
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

    "Refinement criterium"
    ref::R

    "Safety factor below dt limits"
    dt_safety_factor::T
end


"""
Compute derivatives starting from densities ne, ni and stores them in dne, dnh.
"""
function derivs!(dne, dnh, ne, nh, t, fld::StreamerFields, conf::StreamerConf{T},
                 tree, conn, fmgiter=4) where {T}
    (;q, q1, r, q, u, u1, e, flux, eabs, maxdt) = fld
    (;eb, pbc, h, trans, chem, geom, lpl, pbc, fbc) = conf
    
    netcharge!(tree, q, ne, nh, chem)

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

    resize!(maxdt, max(length(maxdt), nblocks(tree)))
    flux!(tree, flux, ne, e, eabs, h, trans, maxdt)
    restrict_flux!(flux, conn)
    #correct_flux!(flux, conn)
    block_derivs!(tree, dne, dnh, ne, nh, fld, conf)
end


"""
Compute the derivatives in a block/tree and store them in dne and dnh.
"""
@bkernel function block_derivs!((tree, level, blkpos, blk), dne, dnh, ne, nh,
                                fld::StreamerFields, conf::StreamerConf)
    
    (;e, flux, eabs) = fld
    (;h, trans, chem, geom) = conf
    isleaf(tree, level, blkpos) || return
    
    chemderivs!((tree, level, blkpos, blk), dne, dnh, ne, nh, eabs, chem, Val{true}())    
    fluxderivs!((tree, level, blkpos, blk), dne, flux, h, geom)

    nothing
end


@bkernel function netcharge!((tree, level, blkpos, blk), q, ne, nh, chem)
    for I in validindices(ne)
        q[I, blk] = netcharge(chem, ne[I, blk], nh[I, blk])
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
function step!(fld::StreamerFields, conf::StreamerConf{T}, tree, conn, t, tmax, ::Val{:ssprk3}) where T
    (;ne, nh, ne1, nh1, ke, kh, maxdt) = fld
    (;dt_safety_factor) = conf

    maxdt .= typemax(eltype(maxdt))
    
    ## This is SSPRK (3rd order SSP Runge-Kutta).
    derivs!(ke[1], kh[1], ne, nh, t, fld, conf, tree, conn)

    # Compute the dt
    dt = min(tmax - t, dt_safety_factor * mapreduce((lvl, I, blk) -> maxdt[blk], min, tree, typemax(T)))
    SBF = identity #StrippedBlockField
    
    ne1 .= ne .+ ke[1] .* dt
    nh1 .= nh .+ kh[1] .* dt
    
    derivs!(ke[2], kh[2], ne1, nh1, t + dt, fld, conf, tree, conn)
    dt1 = dt / 4

    # Here and below there's some bug in VectorOfArray that adds a lot of overhead in broadcasting
    # if I remove the brackets.
    ne1 .= ne .+ (ke[1] .* dt1 .+ ke[2] .* dt1)
    nh1 .= nh .+ (kh[1] .* dt1 .+ kh[2] .* dt1)

    derivs!(ke[3], kh[3], ne1, nh1, t + dt / 2, fld, conf, tree, conn)
    ne .= ne .+ ((ke[1] .* (dt / 6) .+ ke[2] .* (dt / 6)) .+ ke[3] .* (2 * dt / 3))
    nh .= nh .+ ((kh[1] .* (dt / 6) .+ kh[2] .* (dt / 6)) .+ kh[3] .* (2 * dt / 3))

    restrict_full!(ne, conn)
    restrict_full!(nh, conn)

    return (t + dt, dt)
end

function step!(fld::StreamerFields, conf::StreamerConf{T}, tree, conn, t, dt, ::Val{:euler}) where T
    (;ne, nh, ne1, nh1, ke, kh) = fld
    
    derivs!(ke[1], kh[1], ne, nh, t, fld, conf, tree, conn)
    ne .= ne .+ ke[1] .* dt
    nh .= nh .+ kh[1] .* dt
end


function refine!(fld::StreamerFields, conf::StreamerConf, tree, conn, t, freeblocks;
                 minlevel=1, maxlevel=typemax(Int), ref=nothing)
    (;m, refdelta, ne, nh) = fld
    (;h, fbc) = conf

    ref = isnothing(ref) ? conf.ref : ref

    refmark!(tree, m, ref, h, t)
    fill_ghost_copy!(m, conn)
    refdelta!(tree, refdelta, m, BoxStencil{2}())

    fill_ghost_copy!(ne, conn)
    fill_ghost_copy!(nh, conn)

    fill_ghost_bnd!(ne, conn, fbc)
    fill_ghost_bnd!(nh, conn, fbc)

    fill_ghost_interp!(ne, conn)
    fill_ghost_interp!(nh, conn)
    
    applydelta!(tree, refdelta, freeblocks, fld, minlevel, maxlevel)
end
