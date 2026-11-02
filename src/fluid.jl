#=
Routines for solving the fluid equations.
=#

"""
Compute the derivatives resulting from the flux of electrons.
"""
@bkernel function fluxderivs!((tree, level, blkpos, blk),
                              dne::ScalarBlockField{D, M, G, T},
                              flux::VectorBlockField{D, M, G},
                              h, geom) where {D, M, G, T}
    isleaf(tree, level, blkpos) || return

    gbl0 = global_first(blkpos, M)
    h /= 1 << (level - 1)
    
    @inbounds for d in 1:D
        for I in validindices(flux, d)
            I1 = Base.setindex(I, I[d] - 1, d)

            # Global index
            J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
            J1 = Base.setindex(J, J[d] - 1, d)
            
            F = flux[I, d, blk]
            dne[I, blk]  += factor(geom, J, I1 - I, T) * F / h
            dne[I1, blk] -= factor(geom, J1, I - I1, T) * F / h 
        end
    end    
end

abstract type AbstractFluxScheme; end
struct FluxSchemeKoren <: AbstractFluxScheme; end
struct FluxSchemeWENO <: AbstractFluxScheme; end

# Return the number of ghost cells required for a given flux scheme
needghost(::FluxSchemeKoren) = 2
needghost(::FluxSchemeWENO) = 4


"""
Time-step limits at a face: `(advection-diffusion limit, dielectric relaxation time)`.
They are kept apart because only the first one is scaled by the SSP coefficient of the stepper.
"""
@inline function face_maxdt(::Val{D}, h, v, diff, μ, nmax) where {D}
    rate = D * abs(v) / h + 2 * D * diff / h^2
    tmaxwell = convert(typeof(rate), co.epsilon_0 / co.elementary_charge) / (μ * nmax)
    return (inv(rate), tmaxwell)
end


# Componentwise min of two time-step limits, rounded to the element type `S` of maxdt.
@inline _mindt(::Type{S}, a, b) where {S} = map((x, y) -> convert(S, min(x, y)), a, b)


"""
Compute electron fluxes.
"""
@bkernel function flux!((tree, level, blkpos, blk), schem::FluxSchemeKoren,
                        flux::VectorBlockField{D, M, G},
                        ne::ScalarBlockField{D, M, G},
                        e::VectorBlockField{D, M, G},
                        eabs::ScalarBlockField{D, M, G},
                        h, trans, dens, maxdt) where {D, M, G}
    isleaf(tree, level, blkpos) || return

    gbl0 = global_first(blkpos, M)
    h /= 1 << (level - 1)
    # Local accumulator; the convert reproduces the rounding of storing into maxdt at every face.
    S = eltype(eltype(maxdt))
    mdt = (convert(S, Inf), convert(S, Inf))
    
    @inbounds for d in 1:D
        for I in validindices(e, d)
            # Compute flux between I and I1, which is 1 cell lower in the d dimension.
            I1 = Base.setindex(I, I[d] - 1, d)
            x = ntuple(d1 -> h * (gbl0[d1] + I[d1] - G - 1) - (d == d1 ? h : h / 2), Val(D))
            theta = nscale(dens, x)
            
            ed = e[I, d, blk]

            eabs1 = (eabs[I, blk] + eabs[I1, blk]) / 2
            μ = mobility(trans, eabs1 / theta) / theta
            diff = diffusion(trans, eabs1 / theta) / theta
            
            v = -ed * μ
            sv = signbit(v) ? -1 : 1
            
            # The downstream, upstream and twice-upstream indices
            Id  = Base.setindex(I, I[d] + div(sv - 1, 2), d)
            Iu  = Base.setindex(I, I[d] + div(-sv - 1, 2), d)
            Iu2 = Base.setindex(I, I[d] + div(-3sv - 1, 2), d)
            
            F = v * (ne[Iu, blk] + koren_limited(ne[Iu, blk] - ne[Iu2, blk],
                                                 ne[Id, blk] - ne[Iu, blk]))            

            # The diffusion flux
            F += diff * (ne[I1, blk] - ne[I, blk]) / h
            flux[I, d, blk] = F

            # abs: tolerate slightly negative densities
            nmax = max(abs(ne[I, blk]), abs(ne[I1, blk]))
            mdt = _mindt(S, mdt, face_maxdt(Val(D), h, v, diff, μ, nmax))
        end
    end
    maxdt[blk] = mdt
end


# The WENO5 finite-volume reconstruction for conservation laws and cell averages (Jiang & Shu 1996)
# cell averages. Rows of WENO_C: the three 3-cell candidate stencils; WENO_GAMMA: linear weights.
const WENO_C = @SMatrix [1/3  -7/6  11/6   0     0;
                         0    -1/6   5/6   1/3   0;
                         0     0     1/3   5/6  -1/6]
const WENO_GAMMA = @SVector [1/10, 6/10, 3/10]


"""
WENO5 reconstruction of the interface value from the five cell averages `uu`,
ordered upwind -> downwind.
"""
@inline function weno5(uu::SVector{5, T}) where {T}
    # Normalize so that ϵ is scale-free and an all-zero stencil gives 0, not 0/0.
    s = maximum(abs, uu)
    iszero(s) && return zero(T)
    u = uu / s

    # Constants in T: Float64 constants would promote the whole reconstruction.
    c1 = T(13/12)
    c2 = T(1/4)
    β = @SVector [c1 * (u[1] - 2u[2] + u[3])^2 + c2 * (u[1] - 4u[2] + 3u[3])^2,
                  c1 * (u[2] - 2u[3] + u[4])^2 + c2 * (u[2] - u[4])^2,
                  c1 * (u[3] - 2u[4] + u[5])^2 + c2 * (3u[3] - 4u[4] + u[5])^2]
    α = SVector{3, T}(WENO_GAMMA) ./ (T(1e-6) .+ β).^2

    return s * dot(α, SMatrix{3, 5, T}(WENO_C) * u) / sum(α)
end


"""
Compute electron fluxes.
"""
@bkernel function flux!((tree, level, blkpos, blk), schem::FluxSchemeWENO,
                        flux::VectorBlockField{D, M, G},
                        ne::ScalarBlockField{D, M, G},
                        e::VectorBlockField{D, M, G},
                        eabs::ScalarBlockField{D, M, G},
                        h, trans, dens, maxdt) where {D, M, G}
    isleaf(tree, level, blkpos) || return

    gbl0 = global_first(blkpos, M)
    h /= 1 << (level - 1)
    @assert G >= 3
    # Local accumulator; the convert reproduces the rounding of storing into maxdt at every face.
    S = eltype(eltype(maxdt))
    mdt = (convert(S, Inf), convert(S, Inf))
    
    @inbounds for d in 1:D
        for I in validindices(e, d)
            # Compute flux between I and I1, which is 1 cell lower in the d dimension.
            I1 = Base.setindex(I, I[d] - 1, d)
            x = ntuple(d1 -> h * (gbl0[d1] + I[d1] - G - 1) - (d == d1 ? h : h / 2), Val(D))
            theta = nscale(dens, x)

            ed = e[I, d, blk]

            eabs1 = (eabs[I, blk] + eabs[I1, blk]) / 2
            μ = mobility(trans, eabs1 / theta) / theta
            diff = diffusion(trans, eabs1 / theta) / theta
            
            v = -ed * μ
            sv = signbit(v) ? -1 : 1
            
            # Five cells from the most upwind to the most downwind of the interface I-1/2.
            # The first upwind cell is I-1 if v > 0 and I if v < 0.
            iu = I[d] - (sv + 1) ÷ 2
            uu = SVector(ntuple(j -> ne[Base.setindex(I, iu + sv * (j - 3), d), blk], Val(5)))

            F = v * weno5(uu)
            
            # The diffusion flux
            F += diff * (ne[I1, blk] - ne[I, blk]) / h
            flux[I, d, blk] = F

            # abs: tolerate slightly negative densities
            nmax = max(abs(ne[I, blk]), abs(ne[I1, blk]))
            mdt = _mindt(S, mdt, face_maxdt(Val(D), h, v, diff, μ, nmax))
        end
    end
    maxdt[blk] = mdt
end


"""
Correct fluxes from a fine grid to ensure that they are the same as in a coarser grid.

* `src` is the array for the coarse grid.
* `isrc` is a `CartesianIndices` containing the indices of the `src` array.
* `dest` is the array for the fine grid.
* `idest` is a `CartesianIndices` containing the indices of the `dest` array.
* `dim` is the dimension perpendicular to the face.
"""
@generated function correct_flux!(dest::AbstractArray{T, D1}, idest::CartesianIndices{D},
                                  src::AbstractArray{T, D1}, isrc::CartesianIndices{D},
                                  dim) where {T, D1, D}
    # Remember that the last dimension of dest/src is the vector component
    @assert D1 == D + 1
    quote
        k = (1 << ($D - 1))

        # Is, Id -> Indexes of source, dest inside the rectangles
        # Js, Jd -> Indexes in the original arrays (e.g. Js = isrc[Is])
        @inbounds for Is in CartesianIndices(isrc)
            Jd0 = @nref $D idest d -> (d != dim ? 2 * Is[d] - 1 : Is[d])
            Js = isrc[Is]

            f = zero(T)
            @nloops $D i d->(d == dim ? (0:0) : (0:1)) begin
                # The last index is dim because, remember, dest and src are vector arrays so we need
                # the component index.                
                f += @nref $D1 dest d -> (d == $D1 ? dim : Jd0[d] + i_d)
            end

            # Coarse flux
            F = (@nref $D1 src d -> (d == $D1 ? dim : Js[d]))
            c = F - f / k
            
            @nloops $D i d->(d == dim ? (0:0) : (0:1)) begin
                (@nref $D1 dest d -> (d == $D1 ? dim : Jd0[d] + i_d)) += c
            end
        end
    end
end


"""
Correct the flux of a fine grid to be the same as the coarser one in a refinement boundary.
"""
function correct_flux!(f::VectorBlockField{D}, v::Vector{RefBoundary{D}}) where {D}
    for edge in v
        (;coarse, fine, face, subblock) = edge
        idest = bndface(f, face)
        isrc1 = bndface(f, -face)
        dim = perpdim(face)
        isrc = CartesianIndices(ntuple(d -> d == dim ? isrc1.indices[d] :
                                       halfrange(isrc1.indices[d], subblock[d]),
                                       Val(D)))
        
        correct_flux!(f[fine], idest, f[coarse], isrc, dim)
    end
end

"""
Correct the flux of a fine grid to be the same as the coarser one in all refinement boundaries.
"""
function correct_flux!(f::VectorBlockField{D}, conn::Connectivity) where {D}
    # For each layer...
    for (lvl, v) in enumerate(conn.refboundary)
        correct_flux!(f, v)
    end
end


"""
Average the `i` component of the vector field `v` around cell `I`.
"""
@inline function avgvector(v, i, I, D)
    x = zero(eltype(v))
    for S in CubeStencil{D, -1}()
        x += v[(I + S), i]
    end

    return x / 2^D
end


""" The Koren limiter. """        
@inline function koren_limiter(theta)
    (theta >= 4.0) && return 1.0
    (theta > 0.4) && return 1.0 / 3.0 + theta / 6.0
    (theta > 0.0) && return theta
    0.0
end

@inline function koren_limited(a, b)
    (a, b) = promote(a, b)
    
    # theta = a / b. We return ψ(θ) * b
    (a * b < 0) && return zero(a)

    (abs(a) > abs(4b)) && return b
    (abs(10a) > abs(4b)) && return b/3 + a/6
    return a    
end
