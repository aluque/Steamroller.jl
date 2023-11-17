module Steamroll

using DelimitedFiles
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
using TerminalLoggers
using ProgressLogging
using Printf
using Dates
using Format
using Base: @propagate_inbounds
using Interpolations
using FFTW
using SpecialFunctions
using NamedTupleTools
using JLD2

const BlockIndex = Int

@template DEFAULT =
    """
    $(SIGNATURES)
    $(DOCSTRING)
    """

include("constants.jl")
const co = constants

include("bracket.jl")
include("lookup.jl")
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
include("helmholtz.jl")
include("multigrid.jl")
include("gasdensity.jl")
include("transport.jl")
include("refinement.jl")
include("fluid.jl")
include("photo.jl")
include("chemise.jl")
include("chemistry.jl")
include("streamer.jl")
include("freebc.jl")
include("initcond.jl")
include("simulate.jl")
include("util.jl")

include("util/lxcat.jl")

end # module
