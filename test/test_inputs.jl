#=
Tests for the input-file machinery: `include` from input files, functions defined in input files
(world age) and `stop_condition`.

Run with `julia --project=. test/test_inputs.jl`.
=#
using Test
using Steamroller
using DelimitedFiles

const allowed = [:T, Steamroller.allowedinps(Steamroller.InputParameters)...]

struct LargeValue
    x::Vector{Float64}
end

@testset "writejl omits large values" begin
    io = IOBuffer()
    Steamroller.writejl(io, (a=1, b=:koren, c=LargeValue(zeros(100_000)), d=[1.0, 2.0]))
    s = String(take!(io))
    @test occursin("a = 1", s)
    @test occursin("b = :koren", s)
    @test occursin("# c omitted (too large)", s)
    @test occursin("d = [1.0, 2.0]", s)
end

@testset "Input files" begin
    mktempdir() do dir
        mkpath(joinpath(dir, "lib", "sub"))
        write(joinpath(dir, "lib", "sub", "inner.jl"), "inner_value = 3\n")
        write(joinpath(dir, "lib", "outer.jl"), """
              include("sub/inner.jl")
              dt_safety_factor = 0.1 * inner_value
              """)
        write(joinpath(dir, "run.in.jl"), """
              knob = 7
              include("lib/outer.jl")
              poisson_iter = knob
              """)

        @testset "include resolves relative to the including file" begin
            inp = Steamroller.read_input(joinpath(dir, "run.in.jl"), allowed)
            @test inp[:dt_safety_factor] ≈ 0.3
            @test inp[:poisson_iter] == 7
            @test inp[:_included] == [joinpath(dir, "lib", "outer.jl"),
                                      joinpath(dir, "lib", "sub", "inner.jl")]
            # Helper variables are not input parameters
            @test !haskey(inp, :knob)
            @test !haskey(inp, :inner_value)
        end

        @testset "functions from input files and stop_condition" begin
            out = joinpath(dir, "out")
            write(joinpath(dir, "short.in.jl"), """
                  include("lib/outer.jl")
                  T = Float64
                  rootsize = (1, 1)
                  L = 1.25e-2
                  refine_min_h = 5e-5
                  derefine_max_h = 4e-4
                  dt_safety_factor = 0.8
                  output = 0:1e-10:2e-9
                  outfolder = $(repr(out))
                  save = false
                  const seen = Float64[]
                  record(t, conf, fields, tree, conn) = push!(seen, t)
                  output_callbacks = (record,)
                  stop_condition = (t, conf, fields, tree, conn) -> t >= 3e-10
                  """)
            @test Steamroller.main([joinpath(dir, "short.in.jl")]) == 0

            emax = readdlm(joinpath(out, "emax.csv"), ',', skipstart=1)
            @test size(emax, 1) == 4
            @test emax[end, 1] ≈ 3e-10

            @test isfile(joinpath(out, "included", "outer.jl"))
            @test isfile(joinpath(out, "included", "inner.jl"))
            @test readlines(joinpath(out, "included", "paths.txt")) ==
                [joinpath(dir, "lib", "outer.jl"), joinpath(dir, "lib", "sub", "inner.jl")]
        end
    end
end
