using DataStructures

export learn!

function learn!(dtree :: DecisionTree, X :: Matrix, Y :: Vector, depth :: Integer, lossfunction)
    dtree.maxdepth = depth
    dtree.lossfunction = lossfunction
    learn!(dtree.rootnode, X, Y, depth)    
end
function learn!(dnode :: DecisionNode, X :: Matrix, Y :: Vector, depth :: Integer)
    # Check inputs
    depth < 0 && error("Negative depth!")
    size(X, 1) == size(Y, 1) || error("Dimensional missmatch between X and Y!")
    # Init variables
    paramindexes = eachindex(X[1, :])
    # Leaf node
    if depth == 0 || entropy(Y) == 0
        dnode.nodetype = :leaf
        # Set decision as most frequent label
        counts = counter(Y)
        keys = [i for i in eachindex(counts)]
        count, index = findmax([counts[i] for i in keys])
        dnode.decision = keys[index]
        dnode.confidence = count / length(Y)
        return
    end
    # Pick optimal split
    minindex = 1
    minval = Inf
    minθ = nothing
    minmask = nothing
    minsymbol = :leaf
    for paramindex in paramindexes
        val, θ, mask, symbol = bestsplit(X[:, paramindex], Y, X[1, paramindex])
        if val < minval
            minindex = paramindex
            minval = val
            minθ = θ
            minmask = mask
            minsymbol = symbol
        end
    end
    dnode.nodetype = minsymbol
    # Case when no split exists that could separate different labels
    if minval == Inf
        dnode.nodetype = :leaf
        # Set decision as most frequent label
        counts = counter(Y)
        keys = [i for i in eachindex(counts)]
        count, index = findmax([counts[i] for i in keys])
        dnode.decision = keys[index]
        dnode.confidence = count / length(Y)
        return
    end
    # Learn optimal split
    dnode.paramindex = minindex
    dnode.θ = minθ
    # Initialize and learn both leafs
    dnode.leftnode = DecisionNode()
    dnode.rightnode = DecisionNode()
    learn!(dnode.leftnode, X[minmask, :], Y[minmask], depth - 1)
    learn!(dnode.rightnode, X[.!minmask, :], Y[.!minmask], depth - 1)
    return
end

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