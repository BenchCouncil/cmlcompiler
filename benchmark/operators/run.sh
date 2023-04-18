#!/bin/bash
test_framework="framework.csv"
test_breakdown="breakdown.csv"
test_optimization="optimization.csv"
test_elimination="elimination.csv"
num_model=1
num_repeat=5
python test_framework.py 0 $test_framework $test_breakdown False
for((j=1;j<$num_repeat;j++));
do
    python test_framework.py 0 $test_framework $test_breakdown False
done
for((i=1;i<$num_model;i++));
do
    for((j=0;j<$num_repeat;j++));
    do
        python test_framework.py $i $test_framework $test_breakdown False
    done
done

#python test_optimization.py $num_repeat $test_optimization $test_elimination
