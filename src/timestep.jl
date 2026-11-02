#=
Time steppers.
=#

abstract type AbstractTimeStepper end

"Forward Euler."
struct ForwardEuler <: AbstractTimeStepper end

"Explicit midpoint: 2nd order, 2 stages."
struct Midpoint <: AbstractTimeStepper end

"Heun's method or SSPRK(2,2): 2nd order, 2 stages, SSP coefficient 1."
struct Heun <: AbstractTimeStepper end

"SSPRK(3,3): 3rd order, 3 stages, SSP coefficient 1."
struct SSPRK3 <: AbstractTimeStepper end

"SSPRK(4,3) of Kraaijevanger / Spiteri & Ruuth: 3rd order, 4 stages, SSP coefficient 2."
struct SSPRK43 <: AbstractTimeStepper end


"""
Required number of stored derivatives.
"""
nderivs(::AbstractTimeStepper) = 1


"""
Required number of density fields.
"""
nstates(::AbstractTimeStepper) = 1
nstates(::ForwardEuler) = 0


"""
SSP coefficient `C`.
"""
sspcoef(::AbstractTimeStepper) = 1
sspcoef(::SSPRK43) = 2

#=
The explicit midpoint method is not SSP.
=#
sspcoef(::Midpoint) = 1


function timestepper(s::Symbol)
    return (s == :euler    ? ForwardEuler() :
            s == :midpoint ? Midpoint() :
            s == :heun     ? Heun() :
            s == :ssprk3   ? SSPRK3() :
            s == :ssprk43  ? SSPRK43() :
            throw(ArgumentError("Unknown time stepper $s")))
end


"""
Stage update `out[k] = (a[1] x[1][k] + a[2] x[2][k] + ...) + (c[1] dn[1][k] + c[2] dn[2][k] + ...)`
for all species `k`. Note that out may be one of the x.
"""
@bkernel function rkcombine!((tree, level, blkpos, blk), out, x, a, dn, c)
    isleaf(tree, level, blkpos) || return
    @inbounds for k in eachindex(out)
        rkcombine1!(out[k], map(f -> f[k], x), a, map(d -> d[k], dn), c, blk)
    end
end

@inline function rkcombine1!(o, x::NTuple{R}, a::NTuple{R},
                             d::NTuple{S}, c::NTuple{S}, blk) where {R, S}
    @inbounds for I in validindices(o)
        base = x[1][I, blk] * a[1]
        for r in 2:R
            base += x[r][I, blk] * a[r]
        end
        inc = d[1][I, blk] * c[1]
        for s in 2:S
            inc += d[s][I, blk] * c[s]
        end
        o[I, blk] = base + inc
    end
end

# A somewhat faster version specialized for contiguous storage.
@inline function rkcombine1!(o::_ScalarBlockFieldA, x::NTuple{R}, a::NTuple{R},
                             d::NTuple{S}, c::NTuple{S}, blk) where {R, S}
    li = LinearIndices(blksizeghost(o))
    off = (blk - 1) * length(li)
    od = o.u.data
    xd = map(f -> f.u.data, x)
    dd = map(f -> f.u.data, d)
    @inbounds for I in validindices(o)
        j = off + li[I]
        base = xd[1][j] * a[1]
        for r in 2:R
            base += xd[r][j] * a[r]
        end
        inc = dd[1][j] * c[1]
        for s in 2:S
            inc += dd[s][j] * c[s]
        end
        od[j] = base + inc
    end
end

"""
Perform a single step of the time integration with `stepper`, ending no later than `tmax`.
Returns `(t + dt, dt)`.

# Arguments
- `fld::StreamerFields`: the fields to be updated.
- `conf::StreamerConf`: the streamer configuration.
- `tree`: a `Tree` instance containing the tree.
- `conn`: connectivity pattern of the tree.
- `t`: current time.
- `tmax`: maximum time to reach in this step.
- `stepper`: an `AbstractTimeStepper`.
"""
function step!(fld::StreamerFields{T}, conf::StreamerConf{T}, tree, conn, t, tmax,
               stepper::AbstractTimeStepper) where {T}
    (;n, n1, dn, maxdt) = fld
    @assert(length(dn) >= nderivs(stepper) && length(n1) >= nstates(stepper),
            "Not enough storage for $stepper")

    fill!(maxdt, (typemax(T), typemax(T)))
    derivs!(dn[1], n, t, fld, conf, tree, conn)

    C = sspcoef(stepper)
    dt = min(tmax - t, conf.dt_safety_factor *
        mapreduce_tree((lvl, I, blk) -> min(C * maxdt[blk][1], maxdt[blk][2]), min, tree, typemax(T)))

    stages!(fld, conf, tree, conn, t, dt, stepper)
    endstep!(fld, conf, tree, conn)

    return (t + dt, dt)
end


function stages!(fld, conf, tree, conn, t, dt, ::ForwardEuler)
    (;n, dn, photo) = fld
    # To keep type stability also for Float32
    o = one(dt)

    addmcphotosrc!(conf.phmodel, photo, dn[1], dt, true)
    rkcombine!(tree, n, (n,), (o,), (dn[1],), (dt,))
end


function stages!(fld, conf, tree, conn, t, dt, ::Midpoint)
    (;n, dn, photo) = fld
    n1 = fld.n1[1]
    o = one(dt)

    rkcombine!(tree, n1, (n,), (o,), (dn[1],), (dt / 2,))
    derivs!(dn[1], n1, t + dt / 2, fld, conf, tree, conn)

    addmcphotosrc!(conf.phmodel, photo, dn[1], dt, true)
    rkcombine!(tree, n, (n,), (o,), (dn[1],), (dt,))
end


#=
Shu-Osher form:
    n1 = n + dt f(n, t)
    n <- n / 2 + n1 / 2 + dt / 2 f(n1, t + dt)
=#
function stages!(fld, conf, tree, conn, t, dt, ::Heun)
    (;n, dn, photo) = fld
    n1 = fld.n1[1]
    o = one(dt)
    half = o / 2

    addmcphotosrc!(conf.phmodel, photo, dn[1], dt / 2, true)
    rkcombine!(tree, n1, (n,), (o,), (dn[1],), (dt,))
    derivs!(dn[1], n1, t + dt, fld, conf, tree, conn)

    addmcphotosrc!(conf.phmodel, photo, dn[1], dt / 2, false)
    rkcombine!(tree, n, (n, n1), (half, half), (dn[1],), (dt / 2,))
end


#=
Shu-Osher form (Shu & Osher 1988):
    n1 = n + dt f(n, t)
    n1 <- 3 / 4 n + 1 / 4 n1 +   dt / 4 f(n1, t + dt)
    n  <- 1 / 3 n + 2 / 3 n1 + 2 dt / 3 f(n1, t + dt / 2)
Butcher weights b = (1 / 6, 1 / 6, 2 / 3).
=#
function stages!(fld, conf, tree, conn, t, dt, ::SSPRK3)
    (;n, dn, photo) = fld
    n1 = fld.n1[1]
    o = one(dt)

    addmcphotosrc!(conf.phmodel, photo, dn[1], dt / 6, true)
    rkcombine!(tree, n1, (n,), (o,), (dn[1],), (dt,))
    derivs!(dn[1], n1, t + dt, fld, conf, tree, conn)

    addmcphotosrc!(conf.phmodel, photo, dn[1], dt / 6, false)
    rkcombine!(tree, n1, (n, n1), (3 * o / 4, o / 4), (dn[1],), (dt / 4,))
    derivs!(dn[1], n1, t + dt / 2, fld, conf, tree, conn)

    addmcphotosrc!(conf.phmodel, photo, dn[1], 2 * dt / 3, false)
    rkcombine!(tree, n, (n, n1), (o / 3, 2 * o / 3), (dn[1],), (2 * dt / 3,))
end


#=
Shu-Osher form (Spiteri & Ruuth 2002):
    n1 =  n                  + dt / 2 f(n, t)
    n1 <- n1                 + dt / 2 f(n1, t + dt / 2)
    n1 <- 2 / 3 n + 1 / 3 n1 + dt / 6 f(n1, t + dt)
    n  <- n1                 + dt / 2 f(n1, t + dt / 2)
Butcher weights b = (1/6, 1/6, 1/6, 1/2).
=#
function stages!(fld, conf, tree, conn, t, dt, ::SSPRK43)
    (;n, dn, photo) = fld
    n1 = fld.n1[1]
    o = one(dt)

    addmcphotosrc!(conf.phmodel, photo, dn[1], dt / 6, true)
    rkcombine!(tree, n1, (n,), (o,), (dn[1],), (dt / 2,))
    derivs!(dn[1], n1, t + dt / 2, fld, conf, tree, conn)

    addmcphotosrc!(conf.phmodel, photo, dn[1], dt / 6, false)
    rkcombine!(tree, n1, (n1,), (o,), (dn[1],), (dt / 2,))
    derivs!(dn[1], n1, t + dt, fld, conf, tree, conn)

    addmcphotosrc!(conf.phmodel, photo, dn[1], dt / 6, false)
    rkcombine!(tree, n1, (n, n1), (2 * o / 3, o / 3), (dn[1],), (dt / 6,))
    derivs!(dn[1], n1, t + dt / 2, fld, conf, tree, conn)

    addmcphotosrc!(conf.phmodel, photo, dn[1], dt / 2, false)
    rkcombine!(tree, n, (n1,), (o,), (dn[1],), (dt / 2,))
end


"""
Finish a step: deposit Monte Carlo photo-ionization; restrict densities, fill ghosts.
"""
function endstep!(fld::StreamerFields{T, K}, conf, tree, conn) where {T, K}
    (;n) = fld
    (;fbc, dens_ghost_interp) = conf

    mcphoto!(tree, conf.phmodel, fld, conf.h, conf.geom)

    lmax = finestlevel(tree)
    for s in 1:K
        restrict_full!(n[s], conn)
        for l in 1:lmax
            fill_ghost!(n[s], l, conn, fbc, dens_ghost_interp)
        end
    end
end
