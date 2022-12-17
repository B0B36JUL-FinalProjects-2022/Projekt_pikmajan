using Revise
using DecisionTrees
using DataFrames
using CSV
using Statistics
using Random
using Plots
using ProgressMeter
pyplot()

path = "D:/projects_julia/jul-project/DecisionTrees/assets/train.csv"

data = csv2matrix(path)
data_X = data[:, [3, 4, 5, 6, 7, 8, 10, 11, 12]]
X = reformatmatrix(
    data_X, 
    [String, String, String, Int, Int, Int, Float32, String, String],
    ["missing", "missing", "missing", -1, -1, -1, -1.0, "missing", "missing"]
)
data_Y = data[:, [2]]
Y = vec(reformatmatrix(data_Y, [Int], [-1]))

dt = DecisionTree()
learn!(dt, X, Y; depth=21)
Y_ = evaluate(dt, X)
mean(Y .!= Y_)

shuffledidxs = shuffle(collect(1:size(X, 1)))
Xshf = X[shuffledidxs, :]
Yshf = Y[shuffledidxs]

trainproportion = 0.8
traincount = round(Integer, trainproportion * size(Xshf, 1))
testcount = size(Xshf, 1) - traincount

Xtrain = X[begin:traincount, :]
Ytrain = Y[begin:traincount]
Xtest = X[traincount+1:end, :]
Ytest = Y[traincount+1:end]

n = 10
depths = collect(1:20)
errordepths = zeros(length(depths))
@showprogress for j in 1:n
    shuffledidxs = shuffle(collect(1:size(X, 1)))
    Xshf = X[shuffledidxs, :]
    Yshf = Y[shuffledidxs]
    Xtrain = X[begin:traincount, :]
    Ytrain = Y[begin:traincount]
    Xtest = X[traincount+1:end, :]
    Ytest = Y[traincount+1:end]
    for i in 1:20
        dt = DecisionTree()
        learn!(dt, Xtrain, Ytrain; depth=depths[i], attributecount=2)
        Ytest_ = evaluate(dt, Xtest)
        err = mean(Ytest .!= Ytest_)

        errordepths[i] += err
    end
end
plot(depths, errordepths / n)
# Best depth: 3, 4

n = 100
attributes = collect(1:9)
errorattributes = zeros(length(attributes))
@showprogress for j in 1:n
    shuffledidxs = shuffle(collect(1:size(X, 1)))
    Xshf = X[shuffledidxs, :]
    Yshf = Y[shuffledidxs]
    Xtrain = X[begin:traincount, :]
    Ytrain = Y[begin:traincount]
    Xtest = X[traincount+1:end, :]
    Ytest = Y[traincount+1:end]
    for i in 1:9
        dt = DecisionTree()
        learn!(dt, Xtrain, Ytrain; depth=10, attributecount=attributes[i])
        Ytest_ = evaluate(dt, Xtest)
        err = mean(Ytest .!= Ytest_)

        errorattributes[i] += err
    end
end
plot(attributes, errorattributes / n)


