using CSV
using DataFrames

export csv2matrix, reformatmatrix

function csv2matrix(path :: String)
    df = CSV.read(path, DataFrame)
    return Matrix(df)
end

function reformatmatrix(X :: Matrix, types :: Vector{DataType}, defaultvalue :: Vector)
    # Check if arguments have same size
    println(size(X, 2))
    size(X, 2) == length(types) == length(defaultvalue) || throw(AssertionError(""))
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
