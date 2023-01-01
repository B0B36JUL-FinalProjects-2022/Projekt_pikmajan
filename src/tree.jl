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

# ========================================
# evaluate

"""
    evaluate(dtree :: DecisionTree, X)

Predicts value corresponding to `X` by evaluating root `DecisionNode` of 
provided `DecisionTree`.
For reasonable results it is neccessary to maintain feature order the same as 
during learning.

# Example
```julia-repl
julia> X = [-1.5  "a"; -1.14 "b"; -0.45 "aa"; 2.5   "aaa"; 27.4  "aaaa"]
5×2 Matrix{Any}:
 -1.5   "a"
 -1.14  "b"
 -0.45  "aa"
  2.5   "aaa"
 27.4   "aaaa"

julia> Y_ = evaluate(dt, X)
5-element Vector{Int64}:
 0
 1
 0
 1
 1
```
"""
evaluate(dtree :: DecisionTree, X) = evaluate(dtree.root_node, X)

# ========================================
# learn!

"""
    learn!(dtree :: DecisionTree, X :: Matrix, Y :: Vector; depth :: Integer = 1000, attribute_count :: Int = size(X, 2))

Builds `DecisionTree` by recursively expanding root `DecisionNode`.
Input matrix `X` with size `n x m` is a collection of `n` samples, each with 
`m` features.
Input vector `Y` with size `n` is vector of classes corresponding to provided 
samples.
Possible features are `Real, AbstractString`.
Split decisions are selected by maximising information gain.
Keyword argument `depth` limits maximal depth of created tree.
Keyword argument `attribute_count` sets the number of randomly sampled 
attributes which are considered while searching for optimal split.

# Example

``` julia-repl
julia> X = [-1.5  "a"; -1.14 "b"; -0.45 "aa"; 2.5 "aaa"; 27.4 "aaaa"]
5×2 Matrix{Any}:
 -1.5   "a"
 -1.14  "b"
 -0.45  "aa"
  2.5   "aaa"
 27.4   "aaaa"

julia> Y = [0, 1, 0, 1, 1]
5-element Vector{Int64}:
 0
 1
 0
 1
 1

julia> dt = DecisionTree()
Decision tree
    Maximal depth: nothing

    Nodes:
        Leaf node:
        Decision: nothing
        Confidence: nothing

julia> learn!(dt, X, Y)

julia> dt
Decision tree
    Maximal depth: 1000

    Nodes:
        Decision node
        Type: stringinequality
        Parameter index: 2
        θ: aa

        Leaf node:
            Decision: 0
            Confidence: 1.0
        Leaf node:
            Decision: 1
            Confidence: 1.0
```
"""
function learn!(dtree :: DecisionTree, X :: Matrix, Y :: Vector,
    ; depth :: Integer = 1000, attribute_count :: Int = size(X, 2))
    dtree.max_depth = depth
    learn!(dtree.root_node, X, Y; depth, attribute_count)    
end

# ========================================
# to_string

"""
    to_string(dtree :: DecisionTree)

Converts `DecisionTree` and its corresponding tree of `DecisionNode`s to string.

# Example

```julia-repl
julia> dt
Decision tree
    Maximal depth: 1000

    Nodes:
        Decision node
        Type: stringinequality
        Parameter index: 2
        θ: aa

        Leaf node:
            Decision: 0
            Confidence: 1.0
        Leaf node:
            Decision: 1
            Confidence: 1.0
```
"""
function to_string(dtree :: DecisionTree)
    node_str = to_string(dtree.root_node; depth=1)
    return """
    Decision tree
        Maximal depth: $(dtree.max_depth)

        Nodes:
            $node_str"""
end

# ========================================
# Base.show

function Base.show(io :: IO, dtree :: DecisionTree)
    print(io, "$(to_string(dtree))")
end
