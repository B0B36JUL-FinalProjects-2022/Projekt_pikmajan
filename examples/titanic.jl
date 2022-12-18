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


n = 10
depths = collect(1:20)
errordepths = zeros(length(depths))
@showprogress for j in 1:n
    Xshf, Yshf = shuffledata(X, Y)
    Xtrain, Ytrain, Xtest, Ytest = splitdata(X, Y)
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
    Xshf, Yshf = shuffledata(X, Y)
    Xtrain, Ytrain, Xtest, Ytest = splitdata(X, Y)
    for i in 1:9
        dt = DecisionTree()
        learn!(dt, Xtrain, Ytrain; depth=10, attributecount=attributes[i])
        Ytest_ = evaluate(dt, Xtest)
        err = mean(Ytest .!= Ytest_)

        errorattributes[i] += err
    end
end
plot(attributes, errorattributes / n)


