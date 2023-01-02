using ProgressMeter

export RandomForest, learn!, evaluate

mutable struct RandomForest
    size :: Integer
    bagging :: Bool
    dtrees

    function RandomForest(size :: Integer)
        dtrees = [DecisionTree() for i in 1:size]
        new(size, false, dtrees)
    end
end

# ========================================
# evaluate

"""
    evaluate(rforest :: RandomForest, X :: Matrix)

Produces class vector of size `n` with classes predicted by `RandomForest`.
Input matrix X of size `n x m` where `n` is number of samples 
and `m` is feature size.

# Example
```julia-repl
julia> X = [-1.5  "a"; -1.14 "b"; -0.45 "aa"; 2.5 "aaa"; 27.4 "aaaa"]
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
evaluate(rforest :: RandomForest, X :: Matrix) = [evaluate(rforest, X[i, :]) 
                                                  for i in 1:size(X, 1)]

"""
    evaluate(rforest :: RandomForest, X :: Vector)

Produces class prediction for single sample by evaluating all `DecisionTree`s
in the `RandomForest` and selecting most often predicted class.

# Example

```julia-repl
julia> X = [-1.5,  "a"]
2-element Vector{Any}:
 -1.5
   "a"

julia> Y_ = evaluate(rf, X)
0

```
"""
function evaluate(rforest :: RandomForest, X :: Vector)
    counts = counter([evaluate(rforest.dtrees[i], X) 
                      for i in eachindex(rforest.dtrees)])
    keys = [i for i in eachindex(counts)]
    count, index = findmax([counts[i] for i in keys])
    return keys[index]
end

# ========================================
# learn!

"""
    learn!(rforest :: RandomForest, X :: Matrix, Y :: Vector; depth :: Integer = 1000, attribute_count :: Int = size(X, 2), bagging :: Bool = false)

Builds up `DecisionTree`s in `RandomForest`.
Arguments are the same as in the learn function of `DecisionTree`.
When keyword argument `bagging` is true, the data bagging is applied 
during learning.
This means that dataset (values X, Y) are randomly resampled with repetition
uniquely for each tree.
Bagging should result reduction of prediction variance.

# Example
```julia-repl
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

julia> rf = RandomForest(10)
Random Forest
    Tree count: 10
    Bagging: false

    Each tree:
        Maximal depth: nothing
        Attribute count: nothing

julia> learn!(rf, X, Y; depth=2, attribute_count=1, bagging=true)

julia> rf
Random Forest
    Tree count: 10
    Bagging: true

    Each tree:
        Maximal depth: 2
        Attribute count: 1

```
"""
function learn!(rforest :: RandomForest, X :: Matrix, Y :: Vector
    ; depth :: Integer = 1000, attribute_count :: Int = size(X, 2), 
      bagging :: Bool = false, show_progress :: Bool = true)
    rforest.bagging = bagging
    if bagging
        idxs = [sample(1:size(X, 1), size(X, 1); replace=true) 
                for i in 1:rforest.size]
        Xs = [X[idx, :] for idx in idxs]
        Ys = [Y[idx] for idx in idxs]
        prg = Progress(rforest.size; enabled=show_progress)
        for i in eachindex(rforest.dtrees)
            learn!(rforest.dtrees[i], Xs[i], Ys[i]; depth, attribute_count)
            next!(prg)
        end
    else
        prg = Progress(rforest.size; enabled=show_progress)
        for i in eachindex(rforest.dtrees)
            learn!(rforest.dtrees[i], X, Y; depth, attribute_count)
            next!(prg)
        end
    end
    return
end

# ========================================
# to_string

"""
    to_string(rforest :: RandomForest)

Converts `RandomForest` to string by displaying its parameters
and parameters of subordinate `DecisionTree`s.

# Example
```julia-repl
julia> rf
Random Forest
    Tree count: 10
    Bagging: true

    Each tree:
        Maximal depth: 2
        Attribute count: 1

```
"""
function to_string(rforest :: RandomForest)
    str_trees = nothing
    if rforest.dtrees !== nothing
        str_trees = """
                Maximal depth: $(rforest.dtrees[1].max_depth)
                        Attribute count: $(rforest.dtrees[1].attribute_count)"""
    end
    return """
    Random Forest
        Tree count: $(rforest.size)
        Bagging: $(rforest.bagging)
        
        Each tree:
            $(str_trees)"""
end

# ========================================
# Base.show

function Base.show(io :: IO, rforest :: RandomForest)
    print(io, "$(to_string(rforest))")
end
