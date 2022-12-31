export DecisionTree, learn!, evaluate

mutable struct DecisionTree
    # Tree parameters, mostly set after learning
    max_depth :: Union{Nothing, Integer}
    # Root node
    root_node :: DecisionNode

    function DecisionTree()
        new(nothing, DecisionNode())
    end
end

evaluate(dtree :: DecisionTree, X) = evaluate(dtree.root_node, X)

function learn!(dtree :: DecisionTree, X :: Matrix, Y :: Vector,
    ; depth :: Integer = 1000, attribute_count :: Int = size(X, 2))
    dtree.max_depth = depth
    learn!(dtree.root_node, X, Y; depth, attribute_count)    
end

function to_string(dtree :: DecisionTree)
    node_str = to_string(dtree.root_node; depth=1)
    return """
    Decision tree
        Maximal depth: $(dtree.max_depth)

        Nodes:
            $node_str"""
end


function Base.show(io :: IO, dtree :: DecisionTree)
    print(io, "$(to_string(dtree))")
end
