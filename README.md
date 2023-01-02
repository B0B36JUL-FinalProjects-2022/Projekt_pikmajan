# DecisionTrees

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/JuliaTeachingCTU/ImageInspector.jl/blob/master/LICENSE)

DecisionTrees package provides simple implementation of 
decision tree, and random forest for classification.

## Instalation

The package is not registered and it can be installed in the following way:

```julia
(@v1.8) pkg> add https://github.com/B0B36JUL-FinalProjects-2022/Projekt_pikmajan
```

## Further description

The implemented models for classification are:

- __Decision tree__: Implemented with struct named `DecisionTree` in file [tree.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_pikmajan/blob/b9a571d5d8ab021c79b715f931d9dfcd8d08293c/src/tree.jl).
It is provided with functions `learn!`, `evaluate`, and `to_string`.
- __Random forest__: Implemented with struct named `RandomForest` in file [forest.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_pikmajan/blob/b9a571d5d8ab021c79b715f931d9dfcd8d08293c/src/forest.jl).
It is provided with functions `learn!`, `evaluate`, and `to_string`.

Package also includes functions for easier work with datasets, 
these are included in file [data.jl](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_pikmajan/blob/b9a571d5d8ab021c79b715f931d9dfcd8d08293c/src/data.jl).

## Usage and examples

Examples are provided in example folder.
General use of package is briefly described in file [examples/usage.ipynb](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_pikmajan/blob/81266e839b22b82c676c7d48cd5c9e003b9ce5ca/examples/usage.ipynb).
Various models implemented in the package are then used for solving
the Titanic classification task in file [examples/titanic.ipynb](https://github.com/B0B36JUL-FinalProjects-2022/Projekt_pikmajan/blob/81266e839b22b82c676c7d48cd5c9e003b9ce5ca/examples/titanic.ipynb).
