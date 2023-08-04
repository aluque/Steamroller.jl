#=
Code to produce a streamer simulation.
=#

"""
Fields required to run a streamer fluid simulation.
"""
struct StreamerFields{S <: ScalarBlockField, SH <: ScalarBlockField,
                      V <: VectorBlockField, M <: ScalarBlockField,
                      R <: ScalarBlockField}
    "Electron density"
    ne::S

    "Derivative of the electron density"
    dne::S

    # Note that nh are of SCALAR type even if they store several numbers per cell.
    # here SCALAR is used in the physicist's sense.
    "Heavy species density"
    nh::SH

    "Derivative of heavy species"
    dnh::SH
    
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
    
    """
    Initialize a `StreamerFields` instance.
    """
    function StreamerFields(D, M, H, T=Float64, flux_same_as_e=false)
        TH = SVector{H, T}
        ne = ScalarBlockField{D, M, 2, T}()
        dne = ScalarBlockField{D, M, 2, T}()
        nh = ScalarBlockField{D, M, 2, TH}()
        dnh = ScalarBlockField{D, M, 2, TH}()
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
        
        return new{
            typeof(ne), typeof(nh), typeof(e), typeof(m), typeof(refdelta)
        }(ne, dne, nh, dnh, q, q1, u, u1, r, eabs, e, flux, m, refdelta, flux_same_as_e)
    end
end


"""
Add a block to each field in the `StreamerFields` set of fields.
"""
function newblock!(sf::StreamerFields)
    (;ne, dne, nh, dnh, q, q1, u, u1, r, eabs, e, flux, m, refdelta, flux_same_as_e) = sf
    
    n = newblock!(ne)
    
    newblock!(dne, n)
    newblock!(nh, n)
    newblock!(dnh, n)
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
end


"""
Compute the chemistry-part of derivatives in a block/tree.
If `init` is Val{true}() also initializes the derivatives in the same pass.
"""
@bkernel function derivs!((tree, level, blkpos, blk),
                          fld::StreamerFields, conf::StreamerConf)
    
    (;ne, dne, nh, dnh, e, flux, eabs) = fld
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
function step!(fld::StreamerFields, conf::StreamerConf{T}, tree, conn, dt, fmgiter=1;
               update=true) where T
    (;q, q1, r, ne, nh, dne, dnh, q, u, u1, e, flux, eabs) = fld
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
                tree, conn, geom, pbc, lpl, nup=3, ndown=3, ntop=1)
        # fmg!(u, q1, r, u1, h^2 * convert(T, co.elementary_charge / co.epsilon_0),
        #      tree, conn, geom, pbc, lpl, nup=3, ndown=3, ntop=1)
    end

    for l in 1:length(tree)
        # fill_ghost!(ne, l, conn, fbc)
        fill_ghost_copy!(ne, conn.neighbor[l])
        fill_ghost_bnd!(ne, conn.boundary[l], fbc)
        fill_ghost_interp!(ne, conn.refboundary[l], InterpCopy())

        fill_ghost!(u, l, conn, pbc)
    end
    
    electric_field!(tree, e, eabs, u, h, eb)

    for l in 1:length(tree)
        fill_ghost!(eabs, l, conn, ExtrapolateConst())
    end
    
    flux!(tree, flux, ne, e, eabs, h, trans)    
    restrict_flux!(flux, conn)
    #correct_flux!(flux, conn)
    derivs!(tree, fld, conf)

    # StrippedBlockFields are zero-cost abstractions to enable broadcasting without avoiding ghost cells
    # for performance.
    ne1 = StrippedBlockField(ne)
    dne1 = StrippedBlockField(dne)
    nh1 = StrippedBlockField(nh)
    dnh1 = StrippedBlockField(dnh)

    if update
        ne1 .+= dt .* dne1
        nh1 .+= dt .* dnh1
    end
    
    restrict_full!(ne, conn)
    restrict_full!(nh, conn)
end


function refine!(fld::StreamerFields, conf::StreamerConf, tree, conn, freeblocks;
                 minlevel=1, maxlevel=typemax(Int), ref=nothing)
    (;m, refdelta, ne, nh) = fld
    (;h, fbc) = conf

    ref = isnothing(ref) ? conf.ref : ref

    refmark!(tree, m, ref, h)
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
