#=
Routines for solving the fluid equations.
=#

"""
Compute the derivatives resulting from the flux of electrons.
"""
@bkernel function fluxderivs!((tree, level, blkpos, blk),
                              dne::ScalarBlockField{D, M, G},
                              flux::VectorBlockField{D, M, G},
                              h, geom) where {D, M, G}
    isleaf(tree, level, blkpos) || return

    gbl0 = global_first(blkpos, M)
    h /= 1 << (level - 1)
    
    for d in 1:D
        for I in validindices(flux, d)
            I1 = Base.setindex(I, I[d] - 1, d)

            # Global index
            J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
            J1 = Base.setindex(J, J[d] - 1, d)
            
            F = flux[I, d, blk]
            dne[I, blk]  += factor(geom, J, I1 - I) * F / h
            dne[I1, blk] -= factor(geom, J1, I - I1) * F / h 
        end
    end    
end


"""
Compute electron fluxes.
"""
@bkernel function flux!((tree, level, blkpos, blk),
                        flux::VectorBlockField{D, M, G},
                        ne::ScalarBlockField{D, M, G},
                        e::VectorBlockField{D, M, G},
                        eabs::ScalarBlockField{D, M, G},
                        h, trans, maxdt) where {D, M, G}
    isleaf(tree, level, blkpos) || return

    h /= 1 << (level - 1)
    
    for d in 1:D
        for I in validindices(e, d)
            # Compute flux between I and I1, which is 1 cell lower in the d dimension.
            I1 = Base.setindex(I, I[d] - 1, d)
                        
            ed = e[I, d, blk]

            eabs1 = (eabs[I, blk] + eabs[I1, blk]) / 2
            μ = mobility(trans, eabs1)
            diff = diffusion(trans, eabs1)
            
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

            # Compute max dt from CFL, diffusion and Maxwell relaxation)
            maxdt[blk] = min(h / abs(v),
                             h^2 / diff / 4,
                             (co.epsilon_0 / co.elementary_charge) * eabs1 / abs(F))
        end
    end    
end


"""
Compute electron fluxes.
"""
@bkernel function flux_weno!((tree, level, blkpos, blk),
                        flux::VectorBlockField{D, M, G},
                        ne::ScalarBlockField{D, M, G},
                        e::VectorBlockField{D, M, G},
                        eabs::ScalarBlockField{D, M, G},
                        h, trans, maxdt) where {D, M, G}
    gbl0 = global_first(blkpos, M)
    h /= 1 << (level - 1)
    @assert G > 2
    
    for d in 1:D
        for I in validindices(e, d)
            # Compute flux between I and I1, which is 1 cell lower in the d dimension.
            I1 = Base.setindex(I, I[d] - 1, d)
                        
            ed = e[I, d, blk]

            eabs1 = (eabs[I, blk] + eabs[I1, blk]) / 2
            μ = mobility(trans, eabs1)
            diff = diffusion(trans, eabs1)
            
            v = -ed * μ
            sv = signbit(v) ? -1 : 1
            
            # We take five cells: three upstream and two downstream of the interface
            stencil = @SVector([I[d] + div(-5sv - 1, 2),
                                I[d] + div(-3sv - 1, 2),
                                I[d] + div(-sv - 1, 2),
                                I[d] + div(sv - 1, 2),
                                I[d] + div(3sv - 1, 2)])
            S = ntuple(j -> Base.setindex(I, stencil[j], d), Val(5))
            uu = SVector(ntuple(j -> ne[S[j], blk], Val(5)))

            u1 = dot(uu, @SVector [3//8, -5//4, 15//8, 0, 0])
            u2 = dot(uu, @SVector [0, -1//8, 3//4, 3//8, 0])
            u3 = dot(uu, @SVector [0, 0, 3//8, 3//4, -1//8])
            
            β1 = (4 * uu[1]^2 - 19 * uu[1] * uu[2] + 25 * uu[2]^2 + 11 * uu[1] * uu[3]
                  - 31 * uu[2] * uu[3] + 10 * uu[3]^2) / 3
            β2 = (4 * uu[2]^2 - 13 * uu[2] * uu[3] + 13 * uu[3]^2 + 5 * uu[2] * uu[4] - 13 * uu[3] * uu[4]
                  + 4 * uu[4]^2) / 3
            β3 = (10 * uu[3]^2 - 31 * uu[3] * uu[4] + 25 * uu[4]^2 + 11 * uu[3] * uu[5] - 19 * uu[4] * uu[5]
                  + 4 * uu[5]^2) / 3
            β = @SVector [β1, β2, β3]
            γ = @SVector [1//16, 5//8, 5//16]
            ϵ = 1e-14
            
            w1 = @. γ / (ϵ + β)^2
            w = w1 ./ sum(w1) 
            uhalf = dot(w, @SVector [u1, u2, u3])
            
            F = v * uhalf
            
            # The diffusion flux
            F += diff * (ne[I1, blk] - ne[I, blk]) / h
            flux[I, d, blk] = F

            # Compute max dt from CFL, diffusion and Maxwell relaxation)
            maxdt[blk] = min(h / abs(v),
                             h^2 / diff / 4,
                             (co.epsilon_0 / co.elementary_charge) * eabs1 / abs(F))
        end
    end    
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
        for Is in CartesianIndices(isrc)
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


__conserv_coeff0(::Val{2}) = 1//2
__conserv_coeff1(::Val{2}) = 1
__conserv_coeff2(::Val{2}) = -1//4

__conserv_coeff0(::Val{3}) = 1//2
__conserv_coeff1(::Val{3}) = 5//4
__conserv_coeff2(::Val{3}) = -1//4


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
