using Revise
using DecisionTrees
using Statistics

# Small example
X = [-1.5  "a"; -1.14 "b"; -0.45 "aa"; 2.5 "aaa"; 27.4 "aaaa"]
Y = [0, 1, 0, 1, 1]

dt = DecisionTree()
learn!(dt, X, Y)
dt
Y_ = evaluate(dt, X)
mean(Y .!= Y_)

rf = RandomForest(100)
learn!(rf, X, Y; depth=4, attribute_count=2, bagging=true)
rf
Y_ = evaluate(rf, X)
mean(Y .== Y_)
