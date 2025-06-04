#=
This is generic code to handle inputs.  The idea is that you define the input parameters
and their default values simply by adding kwargs to your `main` function (or whatever you want to call
it). The you can use `wrap_input(main, inputfile)` to read `inputfile` and update whatever parameters
are contained there. Now inputfile can be either julia (recommended but unsafe if you do not trust
the source) or toml (safe but inconvenient). Note also that TOML cannot handle julia-specific argument
types (anything other than numbers, strings or booleans), which are forced to take default values.

Also functions writejl and writetoml are provided to write the actually used parameters, for
future reference.
=#
module WrapInput; end

using TOML
using Dates
using LibGit2

function wrap_input(f, finput::String; pretty_print=false, write_inputs=true)
    _, ext = splitext(finput)
    if ext == ".jl"
        ntpl = readjl(f, finput)
    elseif ext == ".toml"
        ntpl = readtoml(f, finput)
    end
    
    input = f(;ntpl...)

    if pretty_print
        io = IOBuffer()
        pprint(IOContext(io, :color => true), input)
        @info "Input read from $finput" * "\n" * String(take!(io))
    end

    if write_inputs
        writejl(joinpath(input.outfolder, input.name * ".in.jl"), f, ntpl)
    end
    simulate(input)
end

"""
Read a julia input file and produces a `NamedTuple` with the values defined there that are also
kwargs of the function `f` or fields if `f` is a type.
"""
function readjl(f, input::String)
    @eval WrapInput include(abspath($input))

    p = Pair{Symbol, Any}[]

    input_vars = allowedinps(f)
    
    for sym in names(WrapInput, all=true)
        if sym in input_vars
            push!(p, sym => getfield(WrapInput, sym))
        end
    end

    if :_input in input_vars
        push!(p, :_input => abspath(input))
    end

    if :_date in input_vars
        push!(p, :_date => now())
    end

    if :_git_commit in input_vars
        hash, dirty = git_commit()
        push!(p, :_git_commit => hash)
        push!(p, :_git_dirty => dirty)
    end
        
    return NamedTuple(p)
end

"Find input variable names allowed for `f`."
function allowedinps(f::Function)
    @assert length(methods(f)) == 1 "Ambiguity: more than one method defined for _main"

    method = first(methods(f))
    return Base.kwarg_decl(method)    
end

allowedinps(f::Type) = fieldnames(f)

"""
Read a toml input file and produces a `NamedTuple` with the values defined there that are also
kwargs of the function `f`.
"""
function readtoml(f, input::String)
    params = TOML.parsefile(input)

    p = Pair{Symbol, Any}[]

    input_vars = allowedinps(f)
    for (k, v) in params
        if Symbol(k) in input_vars
            push!(p, Symbol(k) => v)
        end
    end

    if :_input in input_vars
        push!(p, :_input => abspath(input))
    end

    if :_date in input_vars
        push!(p, :_date => now())
    end    

    if :_git_commit in input_vars
        hash, dirty = git_commit()
        push!(p, :_git_commit => hash)
        push!(p, :_git_dirty => dirty)
    end

    return NamedTuple(p)
end


"""
Writes into a file a julia expression with the kwargs of the function `f` contained in params.
Typically params will be obtained from `Base.@locals`.
"""
function writejl(io::IO, f, params)
    expr = quote end
    input_vars = allowedinps(f)

    for (sym, val) in pairs(params)
        if sym in input_vars
            rval = val isa Symbol ? QuoteNode(val) : val
            push!(expr.args, :($sym = $rval))
        end
    end
    print(io, expr)
end


"""
Calls `writejl(io,...)` opening the file with name `fname`.
"""
function writejl(fname::String, args...)
    open(fname, "w") do fout
        writejl(fout, args...)
    end
end


"""
Finds the root directory of a Git repository given a starting directory.
"""
function find_git_root(start_dir::String)
    current_dir = start_dir
    while current_dir != "/"
        if isdir(joinpath(current_dir, ".git"))
            return current_dir
        end
        current_dir = dirname(current_dir)
    end

    return nothing # No Git repository found
end

"""
Returns the latest git commit and a boolean telling whether the repo is dirty.
"""
function git_commit(repo::GitRepo)
    commit = string(LibGit2.GitHash(LibGit2.peel(LibGit2.GitCommit, LibGit2.head(repo))))
    dirty = LibGit2.isdirty(repo)

    return commit, dirty
end

function git_commit()
    mayberoot = find_git_root(@__DIR__)

    if !isnothing(mayberoot)
        return git_commit(GitRepo(mayberoot))
    else
        return "", true
    end
end



function pprint(io::IO, s, level=0)
    typename = string(typeof(s))
    if length(typename) > 20
        typename = string(Base.typename(typeof(s)).wrapper) * "{…}"
    end
    
    printstyled(io, typename, color=:light_green)
    print(io, "\n")
    for k in fieldnames(typeof(s))
        v = getfield(s, k)
        
        print(io, join(fill(" ", level * 8)))
        printstyled(io, k, color=:light_yellow, bold=true)            
        printstyled(io, " => ", color=:light_black)            
        pprint(io, v, level + 1)
    end
    nothing
end

function pprint(io::IO, s::AbstractVector, level=0)
    print(io, "\n")
    for k in eachindex(s)
        v = getindex(s, k)
        
        print(io, join(fill(" ", level * 8)))
        printstyled(io, "[" * repr(k) * "]", color=:light_yellow, bold=true)            
        printstyled(io, " => ", color=:light_black)            
        pprint(io, v, level + 1)
    end
    nothing
end


function pprint(io::IO, s::Union{Number, AbstractString, AbstractRange, Tuple, Nothing, Symbol, DateTime}, level=0)
    printstyled(io, repr(s) * "\n", color=:blue) 
end


pretty_print(d::Dict, kw...) = pretty_print(stdout, d, kw...)
