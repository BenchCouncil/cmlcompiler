#!/bin/bash
test_framework="233"
test_breakdown="2333"
num_repeat=5
for((j=0;j<$num_repeat;j++));
do
    python test_latency.py $1 $test_framework $test_breakdown False
done
