#=
Implementation of free boundary conditions following Malagon-Romero 2018

Only implemented for cylindrical coordinates.
=#

struct FreeBC{D, M, T, S <: ScalarBlockField, GT, P10, P01}
    geom::GT

    # homogeneous electrostatic Potential
    # we allocate and handle a separate field for this because it is inefficient to alternate
    # between Poisson solutions as we lose information on the previous timestep.
    uh::S

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
    
    function FreeBC(geom::CylindricalGeometry{dim}, u::ScalarBlockField{D, M, G, T},
                    level, rootsize, h) where {dim, D, M, G, T}
        otherdim = mod1(dim + 1, D)
        n = rootsize[otherdim] * 2^(level - 1) * M

        H = rootsize[otherdim] * h * M
        R = rootsize[dim] * h * M
        
        k = (π / H) .* (1:n)
        x = k .* R
        f = @. (2 / H) / (k * (besselix(1, x) / besselix(0, x) +
                               besselkx(1, x) / besselkx(0, x)))
        uh = ScalarBlockField{D, M, G, T}(storage(u))
        
        rodft10 = FFTW.plan_r2r!(f, FFTW.RODFT10)
        rodft01 = FFTW.plan_r2r!(f, FFTW.RODFT01)
        
        new{D, M, T, typeof(uh), typeof(geom), typeof(rodft10), typeof(rodft01)}(geom, uh, H, R, level,
                                                                                 zeros(n),
                                                                                 zeros(n),
                                                                                 zeros(n),
                                                                                 f, rodft10, rodft01)
    end
end


_perpdim(::FreeBC{D, M, T, S, CylindricalGeometry{dim}}) where {dim, D, M, T, S} = dim

function setfreebc!(fr::FreeBC, pbc, strfields::StreamerFields, strconf::StreamerConf{T},
                   tree, conn) where T
    (;uh) = fr
    (;q, q1, r, q, u, u1) = strfields
    (;pbc, h, geom, lpl, poisson_fmg, poisson_iter, poisson_level_iter) = strconf
    (nup, ndown, ntop) = poisson_level_iter
    dim = _perpdim(fr)
    
    for i in 1:poisson_iter
        for l in 1:length(tree)
            fill_ghost!(uh, l, conn, pbc)
        end

        if poisson_fmg
            fmg!(uh, q1, r, u1, h^2 * convert(T, co.elementary_charge / co.epsilon_0),
                 tree, conn, geom, pbc, lpl; nup, ndown, ntop)
        else
            vcycle!(uh, q1, r, u1, h^2 * convert(T, co.elementary_charge / co.epsilon_0),
                    tree, conn, geom, pbc, lpl; nup, ndown, ntop)
        end
    end

    freebc!(q1, uh, h, h^2 * convert(T, co.elementary_charge / co.epsilon_0), tree, conn, fr)
end

setfreebc!(::Nothing, pbc, strfields, strconf, tree, conn) = pbc

function freebc!(q::ScalarBlockField{D, M, G}, u::ScalarBlockField{D, M, G}, h, s, 
                 tree, conn, conf::FreeBC{D}) where {D, M, G}
    @assert D == 2 "Free boundary conditions only support D=2 at the moment"
    (;geom, H, R, level, u1, am, b, f, rodft01, rodft10) = conf

    @assert(finestlevel(tree) <= level,
            "Free b.c. at a level (=$(finestlevel(tree))) higher than the chosen level (=$level) is not implemented")

    dim = _perpdim(conf)
    
    otherdim = mod1(dim + 1, D)
    domain = tree[1].domain

    getouter!(u1, u, tree, dim, level, h)

    mul!(u1, rodft10, u1)
    # This is to make it compatible with the r2r definition of FFTW
    am[end] *= 3
    @. am = f * u1    
    
    mul!(am, rodft01, am)
    @. b = am * h / 2^(level - 1) / 2

    setouter!(q, b, tree, conn, dim, level, s, geom)
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
function setouter!(q::ScalarBlockField{D, M, G}, b, tree, conn, dim, level, s, geom) where {D, M, G}
    otherdim = mod1(dim + 1, D)

    δ = CartesianIndex(ntuple(d -> d == dim ? 1 : 0, Val(D)))
    face0 = Base.setindex(zero(CartesianIndex{D}()), 1, dim)
    
    for l in 1:level
        sl = s / (1 << (l - 1))^2
        p = 2^(level - l)
        
        for blink in conn.boundary[l]
            face0 == blink.face || continue
            k = last(axes(tree[l].domain, dim))
            iblk = blink.block
            B = tree[l].coord[iblk]

            i0 = (B[otherdim] - 1) * M

            for i in 1:M
                I = CartesianIndex(ntuple(d -> d == dim ? G + M : G + i, Val(D)))
                J = CartesianIndex(ntuple(d -> d == dim ? M * (B[d] - 1) + M : M * (B[d] - 1) + i,
                                          Val(D)))
                # Average b
                bmean = sum(@view b[p * (i0 + i - 1) + 1: p * (i0 + i)]) / p
                q[I, iblk] += 2 * factor(geom, J, δ) * bmean / sl
            end
        end

        # We need an extra correction for corners in refinement boundaries. This is because
        # computing the laplacian in the corner includes one ghost cell that is interpolated from
        # a coarser block including one ghost in the coarser mesh. That ghost is not applying the
        # inhomogeneous b.c. so its effect has to be corrected with the source in the corner
        # (non-ghost) cell of the fine grid.
        for rlink in conn.refboundary[l]
            k = last(axes(tree[l].domain, dim))
            iblk = rlink.fine
            B = tree[l].coord[iblk]

            # Not in a boundary
            B[dim] == k || continue
            
            # +1 if fine is below coarse, -1 if fine is above coarse
            dir = rlink.face[otherdim]

            # index in the block that we will affect
            i = dir == -1 ? G + 1 : M + G
            I = CartesianIndex(ntuple(d -> d == dim ? G + M : i, Val(D)))
            
            # Global mesh index
            j = (B[otherdim] - 1) * M + i - G
            J = CartesianIndex(ntuple(d -> d == dim ? M * (B[d] - 1) + M : j, Val(D)))

            J1 = CartesianIndex(ntuple(d -> d == dim ? div(J[d] - 1, 2) + 1 : div(J[d] - 1, 2) + 1 + dir,
                                       Val(D)))
            if dir == 1
                bmean = sum(@view b[p * j + 1 : p * (j + 2)]) / (2p)
            else
                bmean = sum(@view b[p * (j - 3) : p * (j - 1) - 1]) / (2p)
            end
            
            # __w1 is the interpolation weight defined in interp.jl.
            q[I, iblk] += 2 * __w1(Val(D)) * factor(geom, J1, δ) * bmean / sl
        end        
    end
end


function fill_ghost_free!(u::ScalarBlockField{D, M, G}, conf::FreeBC{D}, tree, l) where {D, M, G}
    (;geom, level, b) = conf
    dim = _perpdim(conf)
    otherdim = mod1(dim + 1, D)

    δ = CartesianIndex(ntuple(d -> d == dim ? 1 : 0, Val(D)))
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

fill_ghost_free!(u, ::Nothing, tree, l) = nothing

function newblock!(f::FreeBC, j0=nothing)
    (;uh) = f
    j = newblock!(uh)
    uh[j] .= 0
    !isnothing(j0) && @assert(j0 == j)
end

function interp!(f::FreeBC, dest, src, sub)
    (;uh) = f
    interp!(uh[dest], validindices(uh), uh[src], subblockindices(uh, sub))
end

restrict!(f::FreeBC, dest, src, sub) = nothing

