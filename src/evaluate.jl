export evaluate

evaluate(type :: Symbol, θ :: Real, x :: Real) = x <= θ
evaluate(type :: Symbol, θ :: Nothing, x :: Bool) = x
function evaluate(type :: Symbol, θ :: AbstractString, x :: AbstractString) 
    if type == :stringequality
        return x == θ
    end
    if type == :stringinequality
        return x <= θ
    end
    throw(ArgumentError("Argument nodetype does not exist!"))
end
