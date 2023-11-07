#=
Implementation of free boundary conditions following Malagon-Romero 2018

Only implemented for cylindrical coordinates.
=#

struct FreeBC{D, M, T, GT, P10, P01}
    geom::GT

    # Domain dimensions
    H::T
    R::T
    
    # At what level should we apply free bc?
    level::Int    

    # Workspace
    u1::Vector{T}
    am::Vector{T}
    b::Vector{T}
    f::Vector{T}

    rodft10::P10
    rodft01::P01
    
    function FreeBC{D, M, T}(geom::CylindricalGeometry{dim}, level, rootsize, h) where {dim, D, M, T}
        otherdim = mod1(dim + 1, D)
        n = rootsize[otherdim] * 2^(level - 1) * M

        H = rootsize[otherdim] * h * M
        R = rootsize[dim] * h * M
        
        k = (π / H) .* (1:n)
        x = k .* R
        f = @. (2 / H) / (k * (besselix(1, x) / besselix(0, x) +
                               besselkx(1, x) / besselkx(0, x)))

        rodft10 = FFTW.plan_r2r!(f, FFTW.RODFT10)
        rodft01 = FFTW.plan_r2r!(f, FFTW.RODFT01)
        
        new{D, M, T, typeof(geom), typeof(rodft10), typeof(rodft01)}(geom, H, R, level,
                                                                     zeros(n),
                                                                     zeros(n),
                                                                     zeros(n), f, rodft10, rodft01)
    end
end

_perpdim(::FreeBC{D, M, T, CylindricalGeometry{dim}}) where {dim, D, M, T} = dim

function freebc!(q::ScalarBlockField{D, M, G}, u::ScalarBlockField{D, M, G}, h, s, 
                 tree, conf::FreeBC{D}) where {D, M, G}
    @assert D == 2 "Free boundary conditions only support D=2 at the moment"
    (;geom, H, R, level, u1, am, b, f, rodft01, rodft10) = conf

    @assert(finestlevel(tree) <= level,
            "Free b.c. at a level (=$(finestlevel(tree))) higher than the chosen level (=$level) is not implemented")

    dim = _perpdim(conf)
    
    otherdim = mod1(dim + 1, D)
    domain = tree[1].domain

    getouter!(u1, u, tree, dim, level, h)

    # Do the meaty stuff
    
    mul!(u1, rodft10, u1)
    am[end] *= 3
    @. am = f * u1
    
    # This is to make it compatible with the r2r definition of FFTW
    
    mul!(am, rodft01, am)
    @. b = am * h / 2^(level - 1) / 2
    setouter!(q, b, tree, dim, level, s, geom)
end


"""
Extract the values of the `ScalarBlockField` `u`, which is assumed to be in 2d cylindrical coordinates,
into the vector `u1`, which represents exactly `u` at `level` but has to prolongue/restrict for other
levels. `dim` is the dimension perpendicular to the boundary (i.e. r). Because we use this to obtain
a gradient, we divide by the gridsize of the corresponding level if `h` is not nothing.
"""
function getouter!(u1, u::ScalarBlockField{D, M, G}, tree, dim, level, h=1) where {D, M, G}
    otherdim = mod1(dim + 1, D)

    # Copy blocks into a single 1d array
    for l in 1:level
        k = last(axes(tree[l].domain, dim))
        hl = isnothing(h) ? 1 : h / 2^(l - 1)

        for iblk in axes(tree[l].domain, otherdim)
            B = CartesianIndex(ntuple(d -> d == otherdim ? iblk : k, Val(D)))
            hasblock(tree[l], B) || continue
            
            blk = tree[l][B]
            i0 = (iblk - 1) * M

            # Ratio in cells between the target level and l
            p = 2^(level - l)
            for i in 1:M
                I = CartesianIndex(ntuple(d -> d == dim ? G + M : G + i, Val(D)))
                u1[p * (i0 + i - 1) + 1: p * (i0 + i)] .= u[I, blk] / hl
            end
        end
    end
end


"""
Set the outer values of the charge `q` in order to satisfy the inhomogeneous boundary
conditions given by `b`.
"""
function setouter!(q::ScalarBlockField{D, M, G}, b, tree, dim, level, s, geom) where {D, M, G}
    otherdim = mod1(dim + 1, D)

    δ = CartesianIndex(ntuple(d -> d == dim ? 1 : 0, Val(D)))
    
    for l in 1:level
        sl = s / (1 << (l - 1))^2
        k = last(axes(tree[l].domain, dim))
        for iblk in axes(tree[l].domain, otherdim)
            B = CartesianIndex(ntuple(d -> d == otherdim ? iblk : k, Val(D)))
            hasblock(tree[l], B) || continue

            i0 = (iblk - 1) * M
            blk = tree[l][B]

            p = 2^(level - l)
            for i in 1:M
                I = CartesianIndex(ntuple(d -> d == dim ? G + M : G + i, Val(D)))
                J = CartesianIndex(ntuple(d -> d == dim ? M * (B[d] - 1) + M : M * (B[d] - 1) + i, Val(D)))
                # Average b
                bmean = sum(@view b[p * (i0 + i - 1) + 1: p * (i0 + i)]) / p
                q[I, blk] += 2 * factor(geom, J, δ) * bmean / sl
            end
        end
    end
end


"""
"""
function fill_ghost_free!(u::ScalarBlockField{D, M, G}, conf::FreeBC{D}, tree) where {D, M, G}
    (;geom, level, b) = conf
    dim = _perpdim(conf)    
    otherdim = mod1(dim + 1, D)

    δ = CartesianIndex(ntuple(d -> d == dim ? 1 : 0, Val(D)))
    for l in 1:level
        k = last(axes(tree[l].domain, dim))
        for iblk in axes(tree[l].domain, otherdim)
            B = CartesianIndex(ntuple(d -> d == otherdim ? iblk : k, Val(D)))
            hasblock(tree[l], B) || continue

            i0 = (iblk - 1) * M
            blk = tree[l][B]

            p = 2^(level - l)
            for i in 1:M
                I = CartesianIndex(ntuple(d -> d == dim ? G + M + 1 : G + i, Val(D)))
                # Average b
                bmean = sum(@view b[p * (i0 + i - 1) + 1: p * (i0 + i)]) / p
                u[I, blk] += 2 * bmean
            end
        end
    end
end
    
