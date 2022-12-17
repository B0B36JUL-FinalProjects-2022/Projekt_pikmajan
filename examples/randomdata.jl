using Revise
using DecisionTrees
using Statistics

# Examples of tree learning with datasets of different size
# when the tree depth is unlimited the resulting tree should overfit and 
# achieve 0.0 error rate.

# Small random dataset
samples = 10
features = 4
X = rand(samples, features)
Y = rand(0:5, samples)

dt = DecisionTree()
learn!(dt, X, Y)
Y_ = evaluate(dt, X)
println("Error rate on small dataset: $(1 - mean(Y .== Y_))")


# Medium random dataset
samples = 100
features = 10
X = rand(samples, features)
Y = rand(0:10, samples)

dt = DecisionTree()
learn!(dt, X, Y)
Y_ = evaluate(dt, X)
println("Error rate on medium dataset: $(1 - mean(Y .== Y_))")


# Large random dataset
println("Large dataset:")
samples = 1000
features = 100
X = rand(samples, features)
Y = rand(0:1, samples)

## "Unlimited" depth
dt = DecisionTree()
learn!(dt, X, Y)
Y_ = evaluate(dt, X)
println("Error rate on large dataset: $(1 - mean(Y .== Y_))")

## Limited depth
dt = DecisionTree()
learn!(dt, X, Y; depth=10)
Y_ = evaluate(dt, X)
println("Error rate on large dataset: $(1 - mean(Y .== Y_))")

