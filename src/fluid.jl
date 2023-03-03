#=
Routines for solving the fluid equations.
=#

@bkernel function derivs!((tree, level, blkpos, blk),
                          dne::ScalarBlockField{D, M, G},
                          ne::ScalarBlockField{D, M, G},
                          e::VectorBlockField{D, M, G},
                          eabs::ScalarBlockField{D, M, G},
                          h, trans, geom) where {D, M, G}
    gbl0 = global_first(blkpos, M)
    h /= 1 << (level - 1)
    dne[blk] .= 0
    
    for d in 1:D
        for I in validindices(e, d)
            # Compute flux between I and I1, which is 1 cell lower in the d dimension.
            I1 = Base.setindex(I, I[d] - 1, d)
                        
            # Global index
            J = CartesianIndex(ntuple(d -> I[d] - G + gbl0[d] - 1, Val(D)))
            J1 = Base.setindex(J, J[d] - 1, d)
            
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

            dne[I, blk]  += factor(geom, J, I1 - I) * F / h
            dne[I1, blk] -= factor(geom, J1, I - I1) * F / h 
        end
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
