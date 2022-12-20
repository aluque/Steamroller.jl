module Steamroll

using Base.Cartesian
using StaticArrays
using DocStringExtensions
using DataStructures
using LoopVectorization
using Polyester
using MacroTools
using RecursiveArrayTools

const BlockIndex = Int

@template DEFAULT =
    """
    $(SIGNATURES)
    $(DOCSTRING)
    """


include("mesh.jl")
include("blocklayer.jl")
include("blockfield.jl")
include("stencil.jl")
include("boundary.jl")
include("connectivity.jl")
include("interp.jl")
include("restrict.jl")
include("geometry.jl")
include("laplacian.jl")
include("multigrid.jl")
include("transport.jl")
include("fluid.jl")
include("util.jl")

end # module
