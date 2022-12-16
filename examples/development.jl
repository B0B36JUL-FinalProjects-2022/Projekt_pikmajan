using Revise
using DecisionTrees

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

evaluate(dt, X)

