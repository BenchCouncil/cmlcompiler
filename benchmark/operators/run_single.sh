#!/bin/bash
test_framework="233"
test_breakdown="2333"
num_repeat=1
for((j=0;j<$num_repeat;j++));
do
    python test_framework.py $1 $test_framework $test_breakdown False
done
