using ProgressMeter

export RandomForest, learn!, evaluate

mutable struct RandomForest
    size :: Integer
    dtrees

    function RandomForest(size :: Integer)
        dtrees = [DecisionTree() for i in 1:size]
        new(size, dtrees)
    end
end

evaluate(rforest :: RandomForest, X :: Matrix) = [evaluate(rforest, X[i, :]) 
                                                  for i in 1:size(X, 1)]
function evaluate(rforest :: RandomForest, X :: Vector)
    counts = counter([evaluate(rforest.dtrees[i], X) 
                      for i in eachindex(rforest.dtrees)])
    keys = [i for i in eachindex(counts)]
    count, index = findmax([counts[i] for i in keys])
    return keys[index]
end


function learn!(rforest :: RandomForest, X :: Matrix, Y :: Vector
    ; depth :: Integer = 1000, attribute_count :: Int = size(X, 2), 
      bagging :: Bool = false)
    if bagging
        idxs = [sample(1:size(X, 1), size(X, 1); replace=true) 
                for i in 1:rforest.size]
        Xs = [X[idx, :] for idx in idxs]
        @showprogress for i in eachindex(rforest.dtrees)
            learn!(rforest.dtrees[i], Xs[i], Y; depth, attribute_count)
        end
    else
        @showprogress for i in eachindex(rforest.dtrees)
            learn!(rforest.dtrees[i], X, Y; depth, attribute_count)
        end
    end
    return
end


