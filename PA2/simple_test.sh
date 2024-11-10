for n in 1 10 32 57 89 100 200 343 597 721 996 1000 1573 2000 3097 5000 10000; do
    echo "Testing with n=$n"
    srun -N 1 --gres=gpu:1 ./benchmark $n
    if [ $? -eq 0 ]
    then
        echo "PASS"
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FAIL: n=$n"
        exit 1
    fi
    echo ""
done
