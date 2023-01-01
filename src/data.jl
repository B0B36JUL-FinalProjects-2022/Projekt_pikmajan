using CSV
using DataFrames
using Random

export csv_to_matrix, reformat_data, shuffle_data, split_data

"""
    csv_to_matrix(path :: String)

Reads data stored in CSV file and tranforms them from dataframe to matrix.
"""
function csv_to_matrix(path :: String)
    df = CSV.read(path, DataFrame)
    return Matrix(df)
end

"""
    reformat_data(X :: Matrix, types :: Vector{DataType}, default_value :: Vector)

Changes types of data columns presented in matrix `X` to be the same type 
as specified in the `types` vector.
`missing` values are filled with corresponding `default_value`.
"""
function reformat_data(X :: Matrix, types :: Vector{DataType}, default_value :: Vector)
    # Check if arguments have same size
    println(size(X, 2))
    size(X, 2) == length(types) == length(default_value) || throw(AssertionError("Wrong argument size!"))
    X_ = X
    for i in 1:size(X, 2)
        missing_mask = X_[:, i] .=== missing
        X_[missing_mask, i] .= default_value[i]
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

"""
    shuffle_data(X :: Matrix, Y :: Vector)

Shuffles provided data. 
Pairs of values in `X` and `Y` stay the same.
"""
function shuffle_data(X :: Matrix, Y :: Vector)
    shuffled_idxs = shuffle(collect(1:size(X, 1)))
    X_shf = X[shuffled_idxs, :]
    Y_shf = Y[shuffled_idxs]
    return X_shf, Y_shf
end

"""
    split_data(X :: Matrix, Y :: Vector; proportion=0.8)

Provided data are split into training and testing datasets.
Pairs of values in `X` and `Y` stay the same.
Keyword argument `proportion` determines the train dataset size relative to
original dataset size.
"""
function split_data(X :: Matrix, Y :: Vector; proportion=0.8)
    train_size = round(Integer, proportion * size(X, 1))
    X_train = X[begin:train_size, :]
    Y_train = Y[begin:train_size, :]
    X_test = X[train_size+1:end, :]
    Y_test = Y[train_size+1:end, :]
    return X_train, vec(Y_train), X_test, vec(Y_test)
end
