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
