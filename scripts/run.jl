@warn "This script is deprecated; invoke as `julia -m Steamroller` or install as an app."

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Steamroller: InputParameters, wrap_input, simulate
input = wrap_input(InputParameters, ARGS[1], pretty_print=true)
simulate(input)
