export DecisionNode

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

function tostring(dnode :: DecisionNode; depth :: Integer = 0)
    str = nothing
    if dnode.nodetype == :leaf
        # Show decision and confidence
        return """
        Leaf node:  
            $("    "^depth)Decision: $(dnode.decision)  
            $("    "^depth)Confidence: $(dnode.confidence)""" 
    end
    # Describe desicion and childrens
    decisiontype = dnode.θ === nothing ? Bool : typeof(dnode.θ)
    leftstr = tostring(dnode.leftnode; depth=depth+1)
    rightstr = tostring(dnode.rightnode; depth=depth+1)
    return """
    Decision node
        $("    "^depth)Type: $decisiontype
        $("    "^depth)Parameter index: $(dnode.paramindex)
        $("    "^depth)θ: $(dnode.θ)
        
        $("    "^depth)$leftstr
        $("    "^depth)$rightstr"""
end

function Base.show(io :: IO, dnode :: DecisionNode)
    print(io, "$(tostring(dnode))")
end
