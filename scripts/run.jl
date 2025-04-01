using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Steamroller: InputParameters, wrap_input
wrap_input(InputParameters, ARGS[1], pretty_print=false)
