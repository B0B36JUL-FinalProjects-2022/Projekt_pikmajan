using CSV
using DataFrames
using Random

export csv2matrix, reformatdata, shuffledata, splitdata

function csv2matrix(path :: String)
    df = CSV.read(path, DataFrame)
    return Matrix(df)
end

function reformatdata(X :: Matrix, types :: Vector{DataType}, defaultvalue :: Vector)
    # Check if arguments have same size
    println(size(X, 2))
    size(X, 2) == length(types) == length(defaultvalue) || throw(AssertionError("Wrong argument size!"))
    X_ = X
    for i in 1:size(X, 2)
        missingmask = X_[:, i] .=== missing
        X_[missingmask, i] .= defaultvalue[i]
        if types[i] <: String
            X_[:, i] .= string.(X[:, i])
        elseif types[i] <: Integer       
            X_[:, i] .= round.(types[i], X[:, i])
        else
            X_[:, i] .= convert.(types[i], X[:, i])
        end
    end
    return X_
end

function shuffledata(X :: Matrix, Y :: Vector)
    shuffledidxs = shuffle(collect(1:size(X, 1)))
    Xshf = X[shuffledidxs, :]
    Yshf = Y[shuffledidxs]
    return Xshf, Yshf
end

function splitdata(X :: Matrix, Y :: Vector; proportion=0.8)
    trainsize = round(Integer, proportion * size(X, 1))
    Xtrain = X[begin:trainsize, :]
    Ytrain = Y[begin:trainsize, :]
    Xtest = X[trainsize+1:end, :]
    Ytest = Y[trainsize+1:end, :]
    return Xtrain, vec(Ytrain), Xtest, vec(Ytest)
end
