export DecisionNode, learn!, evaluate

mutable struct DecisionNode
    nodetype :: Symbol
    # Possible leaf value
    decision
    confidence :: Union{Nothing, Real}
    # Childrens
    leftnode :: Union{Nothing, DecisionNode}
    rightnode :: Union{Nothing, DecisionNode}
    # Decision making parameters
    paramindex :: Union{Nothing, Integer}
    θ :: Union{Nothing, AbstractString, Real, Bool}
    
    function DecisionNode()
        new(:leaf, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end

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

function learn!(dnode :: DecisionNode, X :: Matrix, Y :: Vector,
    ; depth :: Integer = 1000, attributecount :: Int = size(X, 2))
    # Check inputs
    depth < 0 && error("Negative depth!")
    size(X, 1) == size(Y, 1) || error("Dimensional missmatch between X and Y!")
    # Init variables
    paramindexes = collect(eachindex(X[1, :]))
    paramindexes = sample(paramindexes, attributecount; replace=false)
    # Leaf node
    if depth == 0 || entropy(Y) == 0
        dnode.nodetype = :leaf
        # Set decision as most frequent label
        counts = counter(Y)
        keys = [i for i in eachindex(counts)]
        count, index = findmax([counts[i] for i in keys])
        dnode.decision = keys[index]
        dnode.confidence = count / length(Y)
        return
    end
    # Pick optimal split
    minindex = 1
    minval = Inf
    minθ = nothing
    minmask = nothing
    minsymbol = :leaf
    for paramindex in paramindexes
        val, θ, mask, symbol = bestsplit(X[:, paramindex], Y, X[1, paramindex])
        if val < minval
            minindex = paramindex
            minval = val
            minθ = θ
            minmask = mask
            minsymbol = symbol
        end
    end
    dnode.nodetype = minsymbol
    # Case when no split exists that could separate different labels
    if minval == Inf
        dnode.nodetype = :leaf
        # Set decision as most frequent label
        counts = counter(Y)
        keys = [i for i in eachindex(counts)]
        count, index = findmax([counts[i] for i in keys])
        dnode.decision = keys[index]
        dnode.confidence = count / length(Y)
        return
    end
    # Learn optimal split
    dnode.paramindex = minindex
    dnode.θ = minθ
    # Initialize and learn both leafs
    dnode.leftnode = DecisionNode()
    dnode.rightnode = DecisionNode()
    learn!(dnode.leftnode, X[minmask, :], Y[minmask]; depth=depth-1, attributecount)
    learn!(dnode.rightnode, X[.!minmask, :], Y[.!minmask]; depth=depth-1, attributecount)
    return
end

function tostring(dnode :: DecisionNode; depth :: Integer = 0)
    if dnode.nodetype == :leaf
        # Show decision and confidence
        return """
        Leaf node:  
            $("    "^depth)Decision: $(dnode.decision)  
            $("    "^depth)Confidence: $(dnode.confidence)""" 
    end
    # Describe desicion and childrens
    leftstr = tostring(dnode.leftnode; depth=depth+1)
    rightstr = tostring(dnode.rightnode; depth=depth+1)
    return """
    Decision node
        $("    "^depth)Type: $(dnode.nodetype)
        $("    "^depth)Parameter index: $(dnode.paramindex)
        $("    "^depth)θ: $(dnode.θ)
        
        $("    "^depth)$leftstr
        $("    "^depth)$rightstr"""
end

function Base.show(io :: IO, dnode :: DecisionNode)
    print(io, "$(tostring(dnode))")
end
