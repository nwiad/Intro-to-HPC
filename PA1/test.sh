#!/bin/bash

function check_range() {
    local nprocs=$1
    local n=$2
    local block_size=$(( (n + nprocs - 1) / nprocs ))
    local IO_offset=$(( block_size * (nprocs - 1) ))
    local out_of_range=$(( IO_offset >= n ))

    # echo "Number of processers: $nprocs    Number of elements: $n"
    # echo "Block size: $block_size    IO offset: $IO_offset    Out of range: $out_of_range"
    return $out_of_range
}


function test_nprocs() {
    mkdir -p test_res
    local output_file="test_res/test_nprocs.out"
    > $output_file
    for num in 100 1000 10000 100000 1000000 10000000 100000000; do
        for ((n=1; n<=28; n++)); do
            echo "Node: 1  Processers: $n  Elements: $num" >> $output_file
            check_range $n $num
            if [ $? -eq 1 ]
            then
                echo "<OUT OF RANGE>" >> $output_file
                srun -N 1 -n $n ./odd_even_sort $num data/$num.dat >> $output_file
                echo "" >> $output_file
            else
                srun -N 1 -n $n ./odd_even_sort $num data/$num.dat >> $output_file
                echo "" >> $output_file
            fi
        done
        for ((n=29; n<=56; n++)); do
            echo "Node: 2  Processers: $n  Elements: $num" >> $output_file
            check_range $n $num
            if [ $? -eq 1 ]
            then
                echo "OUT OF RANGE" >> $output_file
                srun -N 2 -n $n ./odd_even_sort $num data/$num.dat >> $output_file
                echo "" >> $output_file
            else
                srun -N 2 -n $n ./odd_even_sort $num data/$num.dat >> $output_file
                echo "" >> $output_file
            fi
        done
    done
    python3 strip.py $output_file
}


function test_script() {
    mkdir -p test_res
    local output_file="test_res/test_script.out"
    > $output_file
    for num in 100 1000 10000 100000 1000000 10000000 100000000; do
        echo "Number of elements: $num" >> $output_file
        bash run.sh ./odd_even_sort $num data/$num.dat >> $output_file
        echo "" >> $output_file
    done
    python3 strip.py $output_file
}

function get_input_file() {
    local n=$1
    if [ $n -le 100 ] 
    then
        echo "data/100.dat"
    elif [ $n -le 1000 ] 
    then
        echo "data/1000.dat"
    elif [ $n -le 10000 ] 
    then
        echo "data/10000.dat"
    elif [ $n -le 100000 ] 
    then
        echo "data/100000.dat"
    elif [ $n -le 1000000 ] 
    then
        echo "data/1000000.dat"
    elif [ $n -le 10000000 ] 
    then
        echo "data/10000000.dat"
    elif [ $n -le 100000000 ] 
    then
        echo "data/100000000.dat"
    elif [ $n -le 1000000000 ] 
    then
        echo "/home/course/hpc/users/2021012958/PA1_TEST/my_data/1000000000.dat"
    else
        echo "/home/course/hpc/users/2021012958/PA1_TEST/my_data/2147483647.dat"
    fi
}

function test_random_range() {
    local l_range=$1
    local r_range=$2
    local output_file="test_res/test_rand.out"
    echo "Range: $l_range ~ $r_range" >> $output_file
    local rounds=3
    for ((i=0; i<$rounds; i++)); do
        echo "Round $i" >> $output_file
        n=$(( (RANDOM % ($r_range - $l_range + 1) + $l_range) ))
        echo "Number of elements: $n" >> $output_file
        input_file=$(get_input_file $n)
        # echo "Input file: $input_file" >> $output_file
        bash run.sh ./odd_even_sort $n $input_file >> $output_file
        echo "" >> $output_file
    done
}

function test_rand() {
    mkdir -p test_res
    local output_file="test_res/test_rand.out"
    > $output_file
    test_random_range 1 100
    test_random_range 101 1000
    test_random_range 1001 10000
    test_random_range 10001 100000
    test_random_range 100001 1000000
    test_random_range 1000001 10000000
    test_random_range 10000001 100000000
    test_random_range 100000001 1000000000
    test_random_range 1000000001 2147483647
    python3 strip.py $output_file
}

function test_report() {
    mkdir -p test_res
    local output_file="test_res/test_report.out"
    > $output_file
    local args="./odd_even_sort 100000000 data/100000000.dat"
    echo "Node: 1  Processers: 1  Elements: 100000000" >> $output_file
    srun -N 1 -n 1 $args >> $output_file
    echo "" >> $output_file
    echo "Node: 1  Processers: 2  Elements: 100000000" >> $output_file
    srun -N 1 -n 2 $args >> $output_file
    echo "" >> $output_file
    echo "Node: 1  Processers: 4  Elements: 100000000" >> $output_file
    srun -N 1 -n 4 $args >> $output_file
    echo "" >> $output_file
    echo "Node: 1  Processers: 8  Elements: 100000000" >> $output_file
    srun -N 1 -n 8 $args >> $output_file
    echo "" >> $output_file
    echo "Node: 1  Processers: 16  Elements: 100000000" >> $output_file
    srun -N 1 -n 16 $args >> $output_file
    echo "" >> $output_file
    echo "Node: 2  Processers: 32  Elements: 100000000" >> $output_file
    srun -N 2 -n 32 $args >> $output_file
    echo "" >> $output_file
    python3 strip.py $output_file
}

# test_nprocs

# test_script

# test_rand

# test_report