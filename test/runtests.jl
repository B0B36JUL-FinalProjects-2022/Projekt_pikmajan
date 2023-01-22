using Revise
using DecisionTrees
using Test
using Statistics

@testset "DecisionTrees.jl" begin

    @testset "learn.jl" begin
        
        @testset "entropy" begin
            @test DecisionTrees.entropy([0,0,0,0,0,0]) == 0.0
            @test abs(DecisionTrees.entropy([0,1,0,1,0,1]) - 0.6931) <= 0.0001
            @test abs(DecisionTrees.entropy([0,0,0,1,0,1]) - 0.6365) <= 0.0001    
        end
        
        @testset "information_gain" begin
            @test DecisionTrees.information_gain([0,0,1,1], BitVector([0,0,1,1])) == 0.0
            @test DecisionTrees.information_gain([0,0,1,1], BitVector([1,1,0,0])) == 0.0
            @test abs(DecisionTrees.information_gain([0,0,1,1], BitVector([0,1,0,1])) - 0.6931) <= 0.0001
        end

        @testset "best_split" begin
            @test DecisionTrees.best_split(["a","b","c"], [0, 1, 1], "")[2] == "a"
            @test DecisionTrees.best_split(["a","b","c"], [0, 0, 1], "")[2] == "b"
            @test DecisionTrees.best_split(["a","b","c"], [0, 1, 0], "")[2] == "b"

            @test DecisionTrees.best_split([1,2,3], [0, 0, 1], 0)[2] == 2.5
            @test DecisionTrees.best_split([1.5,2,3], [0, 1, 1], 0)[2] == 1.75

            @test DecisionTrees.best_split([true,true,false], [0, 0, 1], true)[1] == 0.0
            @test abs(DecisionTrees.best_split([true,true,false,false], [0,1,0,1], true)[1] - 0.6931) <= 0.0001
        end
    end

    @testset "node.jl" begin
        X = [-1.5  "a"; -1.14 "b"; -0.45 "aa"; 2.5 "aaa"; 27.4 "aaaa"]
        Y = [0, 1, 0, 1, 1]
        dnode = DecisionNode()
        # Goal dnode tree
        dnode_ = DecisionNode()
        dnode_.node_type = :stringinequality
        dnode_.param_index = 2
        dnode_.θ = "aa"
        dn_left = DecisionNode()
        dn_left.node_type = :leaf
        dn_left.decision = 0
        dn_left.confidence = 1.0
        dn_right = DecisionNode()
        dn_right.node_type = :leaf
        dn_right.decision = 1
        dn_right.confidence = 1.0
        dnode_.left_node = dn_left
        dnode_.right_node = dn_right
        
        @testset "learn!" begin
            learn!(dnode, X, Y)
            @test DecisionTrees.to_string(dnode) == DecisionTrees.to_string(dnode_)
        end

        @testset "evaluate" begin
            @test evaluate(:nothing, 0, 0) == true
            @test evaluate(:nothing, -1, 0) == false
            @test evaluate(:nothing, 1, 0) == true
    
            @test evaluate(:nothing, 0.3, 0.3) == true
            @test evaluate(:nothing, -0.3, 0.3) == false
            @test evaluate(:nothing, 0.6, 0.3) == true
    
            @test evaluate(:nothing, nothing, true) == true
            @test evaluate(:nothing, nothing, false) == false
    
            @test evaluate(:stringequality, "a", "a") == true
            @test evaluate(:stringequality, "a", "b") == false
            @test evaluate(:stringequality, "b", "a") == false
    
            @test evaluate(:stringinequality, "a", "a") == true
            @test evaluate(:stringinequality, "a", "b") == false
            @test evaluate(:stringinequality, "b", "a") == true
            
            Y_ = evaluate(dnode, X)
            Y__ = evaluate(dnode_, X)
            @test Y == Y_ == Y__
        end
    end

    @testset "tree.jl" begin
        # Decision tree should be able to match any randomly generated dataset.
        # Other arguments are based on randomness.
        samples = 100
        features = 10
        @testset for i in 1:10
            X = rand(samples, features)
            Y = rand(0:5, samples)
            dt = DecisionTree()
            learn!(dt, X, Y)
            Y_ = evaluate(dt, X)
            @test mean(Y .!= Y_) == 0.0
        end
    end

    @testset "forest.jl" begin
        # Every random forest with trees deep enought 
        # should be able to match any randomly generated dataset.
        # Other arguments are based on randomness.
        samples = 100
        features = 10
        size = 10
        @testset for i in 1:10
            X = rand(samples, features)
            Y = rand(0:5, samples)
            rf = RandomForest(size)
            learn!(rf, X, Y)
            Y_ = evaluate(rf, X)
            @test mean(Y .!= Y_) == 0.0
        end
    end

    @testset "data.jl" begin
        @testset "reformat_data" begin
            X = Matrix{Any}([1.0 1 true 1; missing missing missing missing])
            X_ = reformat_data(X, [Integer, Float64, String, Bool], [5, 3.2, "a", false])
            X_goal = Matrix{Any}([1 1.0 "1.0" true; 5 3.2 "a" false])
            @test X_ == X_goal
        end

        @testset "split_data" begin
            X = reshape(collect(1:10), 10, 1)
            Y = collect(1:10)
            X1, Y1, X2, Y2 = split_data(X, Y)
            @test X1 == reshape(collect(1:8), 8, 1)
            @test Y1 == collect(1:8)
            @test X2 == reshape(collect(9:10), 2, 1)
            @test Y2 == collect(9:10)
        end
    end
end
