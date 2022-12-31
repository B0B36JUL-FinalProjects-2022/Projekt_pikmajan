export evaluate

"""
    evaluate(type :: Symbol, θ :: Real, x :: Real)

Returns boolean corresponding to condition `x <= θ`.
"""
evaluate(type :: Symbol, θ :: Real, x :: Real) = x <= θ

"""
    evaluate(type :: Symbol, θ :: Nothing, x :: Bool)

Returns boolean corresponding to condition `x`.
"""
evaluate(type :: Symbol, θ :: Nothing, x :: Bool) = x

"""
    evaluate(type :: Symbol, θ :: AbstractString, x :: AbstractString)

Returns boolean corresponding to condition `x == θ` for symbol `:stringequality`
and to condition `x <= θ` for symbol `:stringinequality`
"""
function evaluate(type :: Symbol, θ :: AbstractString, x :: AbstractString) 
    if type == :stringequality
        return x == θ
    end
    if type == :stringinequality
        return x <= θ
    end
    throw(ArgumentError("Argument node_type does not exist!"))
end
