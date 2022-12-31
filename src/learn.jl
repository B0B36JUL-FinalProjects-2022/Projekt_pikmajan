using DataStructures
using StatsBase

"""
    best_split(X, Y, sample :: AbstractString)

Function returns best splitting point for AbstractString feature vector 
based on information gain maximization.
"""
function best_split(X, Y, sample :: AbstractString)
    # Get all possible String values
    strings = unique(X)
    # Go through all possible split functions
    # Find minimum split
    min_ig = Inf
    min_θ = nothing
    min_mask = nothing
    min_symbol = :leaf
    for s in strings
        mask = evaluate.(:stringequality, s, X)
        ig = information_gain(Y, mask)
        if ig < min_ig
            min_ig = ig
            min_θ = s
            min_mask = mask
            min_symbol = :stringequality
        end
        mask = evaluate.(:stringinequality, s, X)
        ig = information_gain(Y, mask)
        if ig < min_ig
            min_ig = ig
            min_θ = s
            min_mask = mask
            min_symbol = :stringinequality
        end
    end
    return min_ig, min_θ, min_mask, min_symbol
end

"""
    best_split(X, Y, sample :: Real)

Function returns best splitting point for Real feature vector 
based on information gain maximization.
"""
function best_split(X, Y, sample :: Real)   
    # Get all real values inbetween all present samples
    sorted_X = sort(X)
    reals = (sorted_X[begin:end-1] .+ sorted_X[begin+1:end]) ./ 2
    # Go through all possible split functions
    min_ig = Inf
    min_θ = nothing
    min_mask = nothing
    for r in reals
        mask = evaluate.(:leaf, r, X)
        ig = information_gain(Y, mask)
        if ig < min_ig
            min_ig = ig
            min_θ = r
            min_mask = mask
        end
    end
    return min_ig, min_θ, min_mask, :real
end

"""
    best_split(X, Y, sample :: Bool)

Function returns best splitting point for Bool feature vector 
based on information gain maximization.
"""
function best_split(X, Y, sample :: Bool)
    mask = evaluate.(:leaf, nothing, X)
    min_ig = information_gain(Y, mask)
    return min_ig, nothing, mask, :bool
end

"""
    entropy(Y)

Computes entropy of given class vector.
"""
function entropy(Y)
    length(Y) == 0 && return Inf
    counts = counter(Y)
    return - sum((
        (counts[k] / length(Y)) * log(counts[k] / length(Y)) 
        for k in eachindex(counts)
    ))
end

"""
    information_gain(Y, mask)

Computes information gain of given class vector corresponding to its division
according to provided boolean mask vector. 
"""
function information_gain(Y, mask)
    Y1 = Y[mask]
    Y2 = Y[.!mask]
    h1 = entropy(Y1)
    h2 = entropy(Y2)
    return (length(Y1) / length(Y)) * h1 + (length(Y2) / length(Y)) * h2  
end
