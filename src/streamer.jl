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

    "Electrostatic Potential"
    u::S

    "Helper electrostatic potential field"
    u1::S
    
    "Electric field magnitude"
    eabs::S

    "Electric field"    
    e::V

    "Refinement marker"
    m::M

    "Refinement delta"
    refdelta::R

    """
    Initialize a `StreamerFields` instance.
    """
    function StreamerFields(D, M, H, T=Float64)
        TH = SVector{H, T}
        ne = ScalarBlockField{D, M, 2, T}()
        dne = ScalarBlockField{D, M, 2, T}()
        nh = ScalarBlockField{D, M, 2, TH}()
        dnh = ScalarBlockField{D, M, 2, TH}()
        u = ScalarBlockField{D, M, 2, T}()
        u1 = ScalarBlockField{D, M, 2, T}()
        e = VectorBlockField{D, M, 2, T}()
        eabs = ScalarBlockField{D, M, 2, T}()
        m = ScalarBlockField{D, M, 3, Bool}()
        refdelta = ScalarBlockField{2, 1, 0, RefDelta}()        
        
        return new{
            typeof(ne), typeof(nh), typeof(e), typeof(m), typeof(refdelta)
        }(ne, dne, nh, dnh, u, u1, eabs, e, m, refdelta)
    end
end


@bkernel function derivs!((tree, level, blkpos, blk),
                          fld::StreamerFields, h, trans, chem, geom)
    (;ne, dne, nh, dnh, eabs) = fld

    chemderivs!((tree, level, blkpos, blk), dne, dnh, ne, nh, eabs, chem, Val{true}())
    fluxerivs!((tree, level, blkpos, blk), dne, ne, eabs, h, trans, geom)
end
