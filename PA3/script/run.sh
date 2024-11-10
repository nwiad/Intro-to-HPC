#!/bin/bash
filename=~/PA3/log/output_$(date +"%H_%M_%S_%m_%d").log
k=$1
dsets=$2
# number of parameters must be 2
if [ $# -ne 2 ]; 
then
    echo "shoulf pass 2 parameters: k, dsets (dsets=0/1/2 for k=32, or dsets=1/2 for k=256)"
    echo "Example 1: bash run.sh 32 0"
    echo "Example 2: bash run.sh 256 1"
    exit 1
fi

# k must be 32 or 256
if [ $k -ne 32 ] && [ $k -ne 256 ]; 
then
    echo "k must be 32 or 256"
    exit 1
fi

# dsets must be 0, 1, 2
if [ $dsets -ne 0 ] && [ $dsets -ne 1 ] && [ $dsets -ne 2 ]; 
then
    echo "dsets must be 0, 1, 2"
    exit 1
fi

# if k=256 and dsets=0, warn that it will exceed time limit
if [ $k -eq 256 ] && [ $dsets -eq 0 ]; 
then
    echo "Error: k=256 and dsets=0 will exceed time limit (2 minutes)"
    exit 1
fi

srun -N 1 --gres=gpu:1 ~/PA3/script/run_all.sh $filename $k $dsets
echo "" >> $filename
echo "======================" >> $filename
echo "k=$k" >> $filename
python3 ~/PA3/script/proc.py -f $filename -k $k >> $filename