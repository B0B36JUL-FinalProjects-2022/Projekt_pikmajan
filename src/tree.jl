export DecisionTree

mutable struct DecisionTree
    # Tree parameters, mostly set after learning
    maxdepth :: Union{Nothing, Integer}
    lossfunction
    # Root node
    rootnode :: DecisionNode

    function DecisionTree()
        new(nothing, nothing, DecisionNode())
    end
end

function tostring(dtree :: DecisionTree)
    nodestr = tostring(dtree.rootnode; depth=1)
    return """
    Decision tree
        Maximal depth: $(dtree.maxdepth)
        Loss function: $(dtree.lossfunction)

        Nodes:
            $nodestr"""
end


function Base.show(io :: IO, dtree :: DecisionTree)
    print(io, "$(tostring(dtree))")
end