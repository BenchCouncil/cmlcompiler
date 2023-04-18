#!/bin/bash
test_framework="rasp_framework.csv"
test_breakdown="rasp_breakdown.csv"
num_model=6
num_repeat=5
python test_raspberrypi.py 0 $test_framework $test_breakdown True
for((j=1;j<$num_repeat;j++));
do
    python test_raspberrypi.py 0 $test_framework $test_breakdown False
done
for((i=1;i<$num_model;i++));
do
    for((j=0;j<$num_repeat;j++));
    do
        python test_raspberrypi.py $i $test_framework $test_breakdown False
    done
done
