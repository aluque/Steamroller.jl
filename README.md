# Steamroller.jl
`Steamroller.jl` is a julia code to simulate streamer discharges. This is work in progress and  documentation is lacking but you can take a look at the `samples/` folder for examples.
## Install
To install:
```julia
julia> add https://github.com/aluque/Steamroller.jl
```

## Start a simulation
```julia
julia> import Steamroller as sr
julia> sr.run_from_input("/path/input.in.jl");
```
where you replace `/path/input.in.jl` to the path to an input file.  In the folder `samples/` you can see example inputs.

## Plot results
To plot, use the `SteamrollerMakie.jl` package. Install with
```julia
julia> add https://github.com/aluque/SteamrollerMakie.jl
```

Plot a `.jld2` output file with
```julia
julia> import SteamrollerMakie as srm
julia> srm.scalartreeplot("/path/xxxxx.jld2", :eabs, boundaries=true)
```
This will show the magnitude of electric field. Replace `:eabs` by `(:n, 1)` to plot electron density
(`(:n, 2)` will display density of species number 2 etc.).

