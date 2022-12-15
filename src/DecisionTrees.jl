module DecisionTrees

export learn, greet

include("./DecisionNode.jl")

greet() = println("Hello!")

learn() = eval(10, 12)

end
