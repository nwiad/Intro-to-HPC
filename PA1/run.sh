#!/bin/bash

i=$2

# if i is between 1 and 100
if [ $i -ge 0 ] && [ $i -le 200 ]
then
    srun -N 1 -n 1 --cpu-bind=none ./wrapper.sh $*
fi
# if i is between 101 and 1000
if [ $i -ge 201 ] && [ $i -le 2000 ]
then
    srun -N 1 -n 2 --cpu-bind=none ./wrapper.sh $*
fi
# if i is between 1001 and 10000
if [ $i -ge 2001 ] && [ $i -le 20000 ]
then
    srun -N 1 -n 10 --cpu-bind=none ./wrapper.sh $*
fi
# if i is between 10001 and 100000
if [ $i -ge 20001 ] && [ $i -le 200000 ]
then
    srun -N 1 -n 28 --cpu-bind=none ./wrapper.sh $*
fi
# if i is between 100001 and 100000000
if [ $i -ge 200001 ]
then
    srun -N 2 -n 56 --cpu-bind=none ./wrapper.sh $*
fi

