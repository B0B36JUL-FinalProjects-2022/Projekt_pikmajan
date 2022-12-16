using Revise
using DecisionTrees

X = [
    -1.5  "a"
    -1.14 "b"
    -0.45 "a"
    2.5   "a"
    27.4  "a"
]
Y = [0, 1, 0, 1, 1]


dn = DecisionNode()

learn!(dn, X, Y, 1)

dn

evaluate(dn, X)
