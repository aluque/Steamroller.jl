module Steamroll

using Base.Cartesian
using StaticArrays
using ElasticArrays
using LinearAlgebra
using DocStringExtensions
using DataStructures
using LoopVectorization
using Polyester
using MacroTools
using RecursiveArrayTools
using Logging
using Dates
using Format
using Base: @propagate_inbounds

const BlockIndex = Int

@template DEFAULT =
    """
    $(SIGNATURES)
    $(DOCSTRING)
    """

include("constants.jl")
const co = constants

include("bracket.jl")
include("mesh.jl")
include("blocklayer.jl")
include("tree.jl")
include("treemacros.jl")
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
include("refinement.jl")
include("fluid.jl")
include("chemistry.jl")
include("streamer.jl")
include("util.jl")

end # module
