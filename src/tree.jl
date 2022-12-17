export DecisionTree, learn!, evaluate

mutable struct DecisionTree
    # Tree parameters, mostly set after learning
    maxdepth :: Union{Nothing, Integer}
    # Root node
    rootnode :: DecisionNode

    function DecisionTree()
        new(nothing, DecisionNode())
    end
end

evaluate(dtree :: DecisionTree, X) = evaluate(dtree.rootnode, X)

function learn!(dtree :: DecisionTree, X :: Matrix, Y :: Vector,
    ; depth :: Integer = 1000, attributecount :: Int = size(X, 2))
    dtree.maxdepth = depth
    learn!(dtree.rootnode, X, Y; depth, attributecount)    
end

function tostring(dtree :: DecisionTree)
    nodestr = tostring(dtree.rootnode; depth=1)
    return """
    Decision tree
        Maximal depth: $(dtree.maxdepth)

        Nodes:
            $nodestr"""
end


function Base.show(io :: IO, dtree :: DecisionTree)
    print(io, "$(tostring(dtree))")
end
