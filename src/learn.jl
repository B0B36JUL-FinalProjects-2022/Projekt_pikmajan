using DataStructures
using StatsBase

function bestsplit(X, Y, sample :: AbstractString)
    # Get all possible String values
    strings = unique(X)
    # Go through all possible split functions
    minig = Inf
    minθ = nothing
    minmask = nothing
    minsymbol = :leaf
    for s in strings
        mask = evaluate.(:stringequality, s, X)
        ig = informationgain(Y, mask)
        if ig < minig
            minig = ig
            minθ = s
            minmask = mask
            minsymbol = :stringequality
        end
        mask = evaluate.(:stringinequality, s, X)
        ig = informationgain(Y, mask)
        if ig < minig
            minig = ig
            minθ = s
            minmask = mask
            minsymbol = :stringinequality
        end
    end
    return minig, minθ, minmask, minsymbol
end
function bestsplit(X, Y, sample :: Real)   
    # Get all possible String values
    sortedX = sort(X)
    reals = (sortedX[begin:end-1] .+ sortedX[begin+1:end]) ./ 2
    # Go through all possible split functions
    minig = Inf
    minθ = nothing
    minmask = nothing
    for r in reals
        mask = evaluate.(:leaf, r, X)
        ig = informationgain(Y, mask)
        if ig < minig
            minig = ig
            minθ = r
            minmask = mask
        end
    end
    return minig, minθ, minmask, :real
end
function bestsplit(X, Y, sample :: Bool)
    mask = evaluate.(:leaf, nothing, X)
    minig = informationgain(Y, mask)
    return minig, nothing, mask, :bool
end


function entropy(Y)
    length(Y) == 0 && return Inf
    counts = counter(Y)
    return - sum((
        (counts[k] / length(Y)) * log(counts[k] / length(Y)) 
        for k in eachindex(counts)
    ))
end

function informationgain(Y, mask)
    Y1 = Y[mask]
    Y2 = Y[.!mask]
    h1 = entropy(Y1)
    h2 = entropy(Y2)
    return (length(Y1) / length(Y)) * h1 + (length(Y2) / length(Y)) * h2
    return 0.0    
end
