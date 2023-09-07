"""
Quick macro to debug the times of start and end of executing expression.
"""
macro bracket(expr)
    str = string(expr)
    start = macroexpand(__module__, :(@info("`" * $str * "`  *{*")))
    fin = macroexpand(__module__, :(@info("`" * $str * "` *}*")))

    quote
        $(start)
        $(expr)
        $(fin)
    end |> esc
end
