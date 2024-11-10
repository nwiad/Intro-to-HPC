# 1000 ~ 10000, every 1000, do 10 random tests

function random_test_in_range() {
    local l_range=$1
    local r_range=$2
    local output_file=$3
    echo "Range: $l_range ~ $r_range" >> $output_file
    local rounds=10
    for ((i=0; i<$rounds; i++)); do
        echo "Round $(($i+1))" >> $output_file
        local n=$(( (RANDOM % ($r_range - $l_range) + $l_range) ))
        echo "Testing with n=$n" >> $output_file
        srun -N 1 --gres=gpu:1 ./benchmark $n >> $output_file
        if [ $? -eq 0 ]
        then
            echo "PASS" >> $output_file
        else
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FAIL: n=$n" >> $output_file
            exit 1
        fi
        echo "" >> $output_file
    done
}

function random_test() {
    mkdir -p test_res
    local t=$(date +%s)
    local output_file="test_res/rand_${t}.out"
    echo "output file: $output_file"
    random_test_in_range 1000 2000 $output_file
    random_test_in_range 2000 3000 $output_file
    random_test_in_range 3000 4000 $output_file
    random_test_in_range 4000 5000 $output_file
    random_test_in_range 5000 6000 $output_file
    random_test_in_range 6000 7000 $output_file
    random_test_in_range 7000 8000 $output_file
    random_test_in_range 8000 9000 $output_file
    random_test_in_range 9000 10000 $output_file
}

random_test