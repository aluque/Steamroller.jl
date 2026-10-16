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

using TOML
using Dates
using LibGit2

function read_input(finput::String, allowed)
    _, ext = splitext(finput)
    if ext == ".jl"
        input = readjl(finput, allowed)
    elseif ext == ".toml"
        input = readtoml(finput, allowed)
    else
        error("Only .jl and .toml input files allowed")
    end
    
    return input
end

"""
Include `path` into the input module `mod`, recording its absolute path in `included`.

As with `Base.include`, a relative `path` is resolved against the directory of the file that is
being included, so nested includes work.
"""
function _include_inp(mod, included, path::AbstractString)
    prev = Base.source_path(nothing)
    fullpath = isnothing(prev) ? abspath(path) : normpath(joinpath(dirname(prev), path))
    push!(included, fullpath)
    Base.include(mod, fullpath)
end

"""
Read a julia input file and produces a `NamedTuple` with the values defined there that are also
kwargs of the function `f` or fields if `f` is a type.

Input files may `include` other files (resolved relative to the including file); their paths are
returned in `_included` so that they can be archived with the simulation output.
"""
function readjl(input::String, allowed)
    M = Module(gensym())
    Core.eval(M, :(using Steamroller))
    Core.eval(M, :(const co = Steamroller.co))

    included = String[]
    Core.eval(M, :(include(path::AbstractString) = $_include_inp($M, $included, path)))

    # Base.include (unlike Core.include) sets the source path that nested includes resolve against.
    Base.include(M, abspath(input))
    
    # The @invokelatest here is needed because the bindings in M are too new to be captured
    # at the time where the method was defined.
    @invokelatest _readjl(input, allowed, M, included)
end

function _readjl(input, allowed, M::Module, included=String[])
    p = Dict{Symbol, Any}()

    for sym in names(M, all=true)
        if sym in allowed
            push!(p, sym => getfield(M, sym))
        end
    end

    if :_input in allowed
        push!(p, :_input => abspath(input))
    end

    if :_date in allowed
        push!(p, :_date => now())
    end

    if :_git_commit in allowed
        hash, dirty = git_commit()
        push!(p, :_git_commit => hash)
        push!(p, :_git_dirty => dirty)
    end

    if :_included in allowed
        push!(p, :_included => unique(included))
    end
        
    return p
end

"Find input variable names allowed for `f`."
function allowedinps(f::Function)    
    @assert length(methods(f)) == 1 "Ambiguity: more than one method defined for _main"

    method = first(methods(f))
    return Base.kwarg_decl(method)    
end

allowedinps(f::Type) = fieldnames(f)

allowedinps(f::AbstractVector{Symbol}) = f


"""
Read a toml input file and produces a `NamedTuple` with the values defined there that are also
kwargs of the function `f`.
"""
function readtoml(input::String, allowed)
    params = TOML.parsefile(input)

    p = Pair{Symbol, Any}[]

    for (k, v) in params
        if Symbol(k) in allowed
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
Write into a file a julia expression with the fields and values of `v`.

Values larger than `maxsize` bytes in memory (e.g. models with large lookup tables) are not
written; a comment with their summary is written instead.
"""
function writejl(io::IO, v; maxsize=1_000)
    expr = quote end
    
    for (sym, val) in pairs(ntfromstruct(v))
        if Base.summarysize(val) > maxsize
            push!(expr.args, Expr(:call, :__omitted__, sym, summary(val)))
            continue
        end
        rval = val isa Symbol ? QuoteNode(val) : val
        push!(expr.args, :($sym = $rval))
    end
    print(io, replace(sprint(print, expr),
                      r"__omitted__\(:?(\w+), (\".*\")\)" => s"# \1 omitted (too large): \2"))
end


"""
Call `writejl(io,...)` opening the file with name `fname`.
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
