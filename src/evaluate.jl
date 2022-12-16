export evaluate

evaluate(dtree :: DecisionTree, X) = evaluate(dtree.rootnode, X)
evaluate(dnode :: DecisionNode, X :: Matrix) = [evaluate(dnode, X[i, :]) 
                                                for i in 1:size(X, 1)]
function evaluate(dnode :: DecisionNode, x :: Vector)
    dnode.nodetype == :leaf && return dnode.decision
    return evaluate(dnode, x[dnode.paramindex]) ? 
           evaluate(dnode.leftnode, x) : evaluate(dnode.rightnode, x)
end

evaluate(dnode :: DecisionNode, x :: Real) = x <= dnode.θ
evaluate(dnode :: DecisionNode, x :: Bool) = x
function evaluate(dnode :: DecisionNode, x :: AbstractString) 
    if dnode.nodetype == :stringequality
        return x == dnode.θ
    end
    if dnode.nodetype == :stringinequality
        return x <= dnode.θ
    end
    throw(ArgumentError("Argument nodetype does not exist!"))
end

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
