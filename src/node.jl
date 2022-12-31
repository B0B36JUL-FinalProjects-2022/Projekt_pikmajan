export DecisionNode, learn!, evaluate

mutable struct DecisionNode
    node_type :: Symbol
    # Possible leaf value
    decision
    confidence :: Union{Nothing, Real}
    # Childrens
    left_node :: Union{Nothing, DecisionNode}
    right_node :: Union{Nothing, DecisionNode}
    # Decision making parameters
    param_index :: Union{Nothing, Integer}
    θ :: Union{Nothing, AbstractString, Real, Bool}
    
    function DecisionNode()
        new(:leaf, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

evaluate(dnode :: DecisionNode, X :: Matrix) = [evaluate(dnode, X[i, :]) 
                                                for i in 1:size(X, 1)]
function evaluate(dnode :: DecisionNode, x :: Vector)
    dnode.node_type == :leaf && return dnode.decision
    return evaluate(dnode, x[dnode.param_index]) ? 
           evaluate(dnode.left_node, x) : evaluate(dnode.right_node, x)
end

evaluate(dnode :: DecisionNode, x :: Real) = x <= dnode.θ
evaluate(dnode :: DecisionNode, x :: Bool) = x
function evaluate(dnode :: DecisionNode, x :: AbstractString) 
    if dnode.node_type == :stringequality
        return x == dnode.θ
    end
    if dnode.node_type == :stringinequality
        return x <= dnode.θ
    end
    throw(ArgumentError("Argument node_type does not exist!"))
end

function learn!(dnode :: DecisionNode, X :: Matrix, Y :: Vector,
    ; depth :: Integer = 1000, attribute_count :: Int = size(X, 2))
    # Check inputs
    depth < 0 && error("Negative depth!")
    size(X, 1) == size(Y, 1) || error("Dimensional missmatch between X and Y!")
    # Init variables
    param_indexes = collect(eachindex(X[1, :]))
    param_indexes = sample(param_indexes, attribute_count; replace=false)
    # Leaf node
    if depth == 0 || entropy(Y) == 0
        dnode.node_type = :leaf
        # Set decision as most frequent label
        counts = counter(Y)
        keys = [i for i in eachindex(counts)]
        count, index = findmax([counts[i] for i in keys])
        dnode.decision = keys[index]
        dnode.confidence = count / length(Y)
        return
    end
    # Pick optimal split
    min_index = 1
    min_val = Inf
    min_θ = nothing
    min_mask = nothing
    min_symbol = :leaf
    for param_index in param_indexes
        val, θ, mask, symbol = best_split(X[:, param_index], Y, X[1, param_index])
        if val < min_val
            min_index = param_index
            min_val = val
            min_θ = θ
            min_mask = mask
            min_symbol = symbol
        end
    end
    dnode.node_type = min_symbol
    # Case when no split exists that could separate different labels
    if min_val == Inf
        dnode.node_type = :leaf
        # Set decision as most frequent label
        counts = counter(Y)
        keys = [i for i in eachindex(counts)]
        count, index = findmax([counts[i] for i in keys])
        dnode.decision = keys[index]
        dnode.confidence = count / length(Y)
        return
    end
    # Learn optimal split
    dnode.param_index = min_index
    dnode.θ = min_θ
    # Initialize and learn both leafs
    dnode.left_node = DecisionNode()
    dnode.right_node = DecisionNode()
    learn!(dnode.left_node, X[min_mask, :], Y[min_mask]; depth=depth-1, attribute_count)
    learn!(dnode.right_node, X[.!min_mask, :], Y[.!min_mask]; depth=depth-1, attribute_count)
    return
end

function to_string(dnode :: DecisionNode; depth :: Integer = 0)
    if dnode.node_type == :leaf
        # Show decision and confidence
        return """
        Leaf node:  
            $("    "^depth)Decision: $(dnode.decision)  
            $("    "^depth)Confidence: $(dnode.confidence)""" 
    end
    # Describe desicion and childrens
    leftstr = to_string(dnode.left_node; depth=depth+1)
    rightstr = to_string(dnode.right_node; depth=depth+1)
    return """
    Decision node
        $("    "^depth)Type: $(dnode.node_type)
        $("    "^depth)Parameter index: $(dnode.param_index)
        $("    "^depth)θ: $(dnode.θ)
        
        $("    "^depth)$leftstr
        $("    "^depth)$rightstr"""
end

function Base.show(io :: IO, dnode :: DecisionNode)
    print(io, "$(to_string(dnode))")
end
