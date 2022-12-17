using Revise
using DecisionTrees
using Statistics

# Small example
X = [
    -1.5  "a"
    -1.14 "b"
    -0.45 "aa"
    2.5   "aaa"
    27.4  "aaaa"
]
Y = [0, 1, 0, 1, 1]

dt = DecisionTree()
learn!(dt, X, Y; depth=1, attributecount=1)
dt
Y_ = evaluate(dt, X)
mean(Y .== Y_)

rf = RandomForest(10)
learn!(rf, X, Y; bagging=true)
Y_ = evaluate(rf, X)
mean(Y .== Y_)
