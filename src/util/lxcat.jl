module LxCatSwarmData

using DataFrames
using CSV

struct LxCatTable
    data::DataFrame
    index::Dict{String, Int}
    explain::Dict{String, String}
end


function load(fname)
    headers = ["Transport coefficients" => "transport",
               "Rate coefficients (m3/s)" => "rates",
               "Inverse rate coefficients (m3/s)" => "inverse_rates"]
    
    tables = Dict(h => String[] for (h0, h) in headers)
    current = nothing
    for line in eachline(fname)
        iheader = findfirst(((h0, h),) -> occursin(h0, line), headers)
        
        if !isnothing(iheader)
            current = headers[iheader][2]
        elseif !isnothing(current)
            push!(tables[current], line)
        end            
    end
    
    (transport, rates, inverse_rates) = [parse_table(tables[k]) for k in
                                         ["transport", "rates", "inverse_rates"]]
    
    
    inverse_rates = rename_inverses(inverse_rates)

    data = innerjoin(transport.data, rates.data, inverse_rates.data,
                     on=[:run, :en], makeunique=true)
    explain = merge(transport.explain, rates.explain, inverse_rates.explain)
    index = Dict(map(reverse, collect(pairs(names(data)))))
    
    return LxCatTable(data, index, explain)
end

"""
Parses a table from a list of lines.
"""
function parse_table(linelist)    
    # explanations
    explain = Dict{String, String}()

    # data
    data = String[]

    for line in linelist
        m = match(r"^\s*([AC]\d+)\s+(.*)", line)
        if !isnothing(m)
            explain[m.captures[1]] = String(strip(m.captures[2]))
        else
            push!(data, strip(line))
        end
    end
    # Avoid spaces or weird chars in column names
    data[1] = replace(data[1],
                      "R#" => "run",
                      "E/N (Td)" => "en",
                      "Energy (eV)" => "energy")
                      
    data = CSV.read(IOBuffer(join(data, "\n")), delim=" ", ignorerepeated=true, DataFrame)
    (;explain, data)
end

function rename_inverses(inverses)
    # Rename the inverse_rates table to be able to join
    replacements = map(name -> name => replace(name, r"C(\d+)" => s"I\1"),
                       names(inverses.data))
    rename!(inverses.data, replacements)
    
    explain = Dict((replace(k, replacements...) => "(inverse) " * v) for (k, v) in inverses.explain)
    data = inverses.data

    (;explain, data)
end


end
