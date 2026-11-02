#= PHOTOIONIZATION
   We basically reuse all the code in multigrid changing the applied stencil.
=#

abstract type PhotoionizationModel; end

"""
A single term in the Helmholtz approximation to photoionization.
"""
struct PhotoionizationTerm{T}
    "Multiplicative factor"
    a::T

    "k^2 in ∇²u - k² u = S"
    k2::T
end

"""
A set of terms representing a full photoionization model with `L` terms, affecting densities of two
species in `target`.
"""
struct FluidPhotoionizationModel{L, T, BC, P} <: PhotoionizationModel
    "Each `PhotoionizationTerm` in the Helmholtz approximation."
    term::SVector{L, PhotoionizationTerm{T}}

    "Species index of the source."
    source::Int
    
    "Target species (generally electrons and the ionized positive ions)"
    target::NTuple{2, Int}    

    "Boundary conditions for the photoionization fields."
    bc::BC

    "Number of vcycles in the solution."
    iter::Int
end

FluidPhotoionizationModel{L, T, BC}(args...) where {L, T, BC} = FluidPhotoionizationModel{L, T, BC, 2}(args...)
_order(::FluidPhotoionizationModel{L, T, BC, P}) where {L, T, BC, P} = P

Base.length(::FluidPhotoionizationModel{L}) where {L} = L

####
##.  Monte Carlo Photoionization
####
struct MontecarloPhotoionizationModel{T, D, M}  <: PhotoionizationModel
    "The efficiency factor (from production of a species S to effective photons."
    eff::T

    "Species index of the source."
    source::Int
    
    "Target species (generally electrons and the ionized positive ions)"
    target::NTuple{2, Int}    

    "The log of the photon absorption rate is uniformly sampled between chimin and chimax."
    logchimin::T

    "The log of the photon absorption rate is uniformly sampled between chimin and chimax."
    logchimax::T
end

Base.length(::MontecarloPhotoionizationModel) = 1


"""
Add the photoionization term to the density derivatives in `dni`.
"""
function photoionization!(tree::Tree{D}, dni, phfields, r, u1,
                          phmodel::FluidPhotoionizationModel, h, conn, geom) where D
    (;term, source, target, bc, iter) = phmodel
    lmax = finestlevel(tree)::Int
    for i in eachindex(term)
        for l in 1:lmax
            fill_ghost!(phfields[i], l, conn, bc)
        end

        helm = HelmholtzDiscretization{D, _order(phmodel), typeof(term[i].k2)}(term[i].k2)
        for j in 1:iter
            vcycle!(phfields[i], dni[source], r, u1, h^2,
                    tree, conn, geom, bc, helm, nup=2, ndown=2, ntop=4)
        end
    end

    # Because the source may be changed, only after computing all fields can we add them to the
    # derivatives.
    for i in eachindex(term)
        a = term[i].a
        add_photoionization_term!(tree, dni, phfields[i], a, phmodel)
    end

end

@bkernel function add_photoionization_term!((tree, level, blkpos, blk), dn, u, a, phmodel)
    isleaf(tree, level, blkpos) || return
    @inbounds for I in validindices(dn[1])
        for k in phmodel.target
            dn[k][I, blk] += a * u[I, blk]
        end
    end        
end

photoionization!(tree::Tree, dni, phfields, r, u1,
                 phmodel::MontecarloPhotoionizationModel, h, conn, geom) = nothing

###
#  Predefined photo-ionization models
##
"""
Construct an empty photo-ionization model.  Because the number of terms (0) is known statically,
this should have a zero overhead.
"""
function empty_photoionization(T)    
    return FluidPhotoionizationModel{0, T, Nothing}(SVector{0, PhotoionizationTerm{T}}(), 0, (0, 0), nothing, 0)    
end


"""
Construct the bourdon 2/3-term model as described by Bagheri 2018.
"""
function bourdon(T, A, λ, D=2;
                 # Index of the source (the species that contains ionization used as a source).
                 source=2,
                 # ξB νu/νi
                 ξ´ = 0.075,
                 pO2 = 150 * co.torr,
                 p = 750 * co.torr,
                 pq = 40 * co.milli * co.bar)

    # See A.9 for the multiplicative factors.  Note also the unit conversion.
    a = @. A * pO2^2 * (pq / (p + pq)) * ξ´ * co.centi^-2 * co.torr^-2
    k = @. pO2 * λ * co.centi^-1 * co.torr^-1

    term = PhotoionizationTerm{T}.(a, k .^ 2)
    bc = (D == 2 ?
        boundaryconditions(((1, 1), (-1, -1))) :
        boundaryconditions(((-1, -1), (-1, -1), (-1, -1))))
    
    phmodel = FluidPhotoionizationModel{length(A), T, typeof(bc)}(term, 1, (1, 2), bc, 2)

    return phmodel
end

# Direct copy of table A2 in Bagheri 2018
bourdon2(T=Float64, D=2) = bourdon(T,
                                   [0.0021, 0.1775], # A
                                   [0.0974, 0.5877], # λ
                                   D)

# Direct copy of table A3 in Bagheri 2018
bourdon3(T=Float64, D=2) = bourdon(T,
                                   [1.986e-4, 0.0051, 0.4886],  # A
                                   [0.0553, 0.1460, 0.8900],    # λ
                                   D)


"""
Accumulate `w * dni[source]` into the Monte Carlo photo-ionization source `photo[1]`, overwriting
it if `init`.  Time steppers call this once per stage with `w = dt * b` (`b` the stage weight), so
at the end of a step `photo[1]` holds the production of the source species during the step.
"""
function addmcphotosrc!(model::MontecarloPhotoionizationModel, photo, dni, w, init)
    s = model.source
    if init
        photo[1] .= dni[s] .* w
    else
        photo[1] .= photo[1] .+ dni[s] .* w
    end
end

addmcphotosrc!(::FluidPhotoionizationModel, photo, dni, w, init) = nothing

"""
Add photo ionization production. The field fields.photo[1] must be filled with the "source" field;
for example in the Zhelezniak model where photon emission is assumed to be proportional to impact
ionization this field must contain the total number density of impact ionization events in a cell.
"""
function mcphoto!((tree, lvl, coord, blk),
                                        model::MontecarloPhotoionizationModel{T, D, M},
                                        fields, h, 
                                        geom::AbstractGeometry) where {T, D, M}
    
    isleaf(tree, lvl, coord) || return
    (;source, target, eff) = model
    (;n, photo) = fields
    hl = unitcell(lvl, h)

    @inbounds for I in localindices(photo[1])
        I1 = addghost(photo[1], I)
        x0 = cell_lower(coord, I, lvl, M, h)
        dv = cellvolume(geom, coord, I, lvl, M, h)

        # Compute number of photons
        nex = dv * photo[1][I1, blk] * eff

        # Poisson(...) errors for negative arguments. For small negative numbers we clamp to zero
        # but if they are large and negative this may indicate a problem elsewhere so we fail here.
        if nex < 0 && nex > -eps(typeof(nex))
            nex = zero(nex)
        end
        
        nphotons = rand(Poisson(nex))
        
        for j in 1:nphotons
            x = ntuple(d -> x0[d] + hl * rand(typeof(hl)), Val(D))
            xabs = samplephoton(x, model, geom, Val(D))
            loc = treelocate(tree, xabs, M, h)
            isnothing(loc) && continue
            (blk1, m1, l1, coord1) = loc
            w = 1 / cellvolume(geom, coord1, m1, l1, M, h)
            for trg in target
                atomicadd!(n[trg], addghost(n[trg], m1), blk1, w)
            end
        end
    end
end

#=
Here we use Threads.@spawn because @batch / @pbatch allocates statically giving roughly the
same number of blocks per thread. Because some blocks (with higher photon production) require much more
work than other that leaves threads underused.
=#
function mcphoto!(tree::Tree, model::MontecarloPhotoionizationModel, fields, h, geom::AbstractGeometry)
    next = Threads.Atomic{Int}(1)
    nb = nblocks(tree)
    @sync for _ in 1:Threads.nthreads()
        Threads.@spawn while (i = Threads.atomic_add!(next, 1)) <= nb
            (lvl, coord, blk) = nth(tree, i)
            mcphoto!((tree, lvl, coord, blk), model, fields, h, geom)
        end
    end
end

mcphoto!(tree, model::FluidPhotoionizationModel, fields, args...) = nothing

"""
Sample the absorption location of a photon emitted at `x`.
"""
function samplephoton(x, model::MontecarloPhotoionizationModel{T}, geom::CartesianGeometry,
                      dim::Val{3}) where T
    (;logchimin, logchimax) = model
    ell = exp(-(logchimin + rand(T) * (logchimax - logchimin)))
    rabs = rand(Exponential(ell))
    return x .+ rabs .* randsphere(T)
end

function samplephoton(x, model::MontecarloPhotoionizationModel{T},
                      geom::CylindricalGeometry{C}, dim::Val{2}) where {T, C}
    (;logchimin, logchimax) = model
    ell = exp(-(logchimin + rand(T) * (logchimax - logchimin)))
    rabs = rand(Exponential(ell))
    x1 = SA[x[1], x[2], 0] .+ rabs .* randsphere(T)
    rho = hypot(x1[C], x1[3])
    return SVector(ntuple(d -> d == C ? rho : x1[d], Val(2)))
end


"""
Sample points from the unitary sphere, with coordinates of type `T`.
"""
function randsphere(::Type{T}=Float64) where T
    ϕ = T(2π) * rand(T)
    sinϕ, cosϕ = sincos(ϕ)
    
    u = 2 * rand(T) - 1
    v = sqrt(1 - u^2)

    @SVector [v * cosϕ, v * sinϕ, u]
end
