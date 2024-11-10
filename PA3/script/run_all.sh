#!/bin/bash
#SBATCH --job-name=PA3_run_all
#SBATCH --output=PA3_run_all_%j.log
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:1

dsets_0=(arxiv collab citation ddi protein ppa reddit.dgl products youtube amazon_cogdl yelp wikikg2 am)
dsets_1=(arxiv collab citation ddi protein ppa reddit.dgl products youtube)
dsets_2=(amazon_cogdl yelp wikikg2 am)

filename=$1
k=$2
dsets=$3

echo Log saved to $filename


if [ $dsets -eq 0 ]; then
    for j in `seq 0 $((${#dsets_0[@]}-1))`;
    do
        echo ${dsets_0[j]}
        ~/PA3/build/test/unit_tests --dataset ${dsets_0[j]}   --len $k --datadir ~/PA3/data/  2>&1 | tee -a $filename 
    done
elif [ $dsets -eq 1 ]; then
    for j in `seq 0 $((${#dsets_1[@]}-1))`;
    do
        echo ${dsets_1[j]}
        ~/PA3/build/test/unit_tests --dataset ${dsets_1[j]}   --len $k --datadir ~/PA3/data/  2>&1 | tee -a $filename 
    done
elif [ $dsets -eq 2 ]; then
    for j in `seq 0 $((${#dsets_2[@]}-1))`;
    do
        echo ${dsets_2[j]}
        ~/PA3/build/test/unit_tests --dataset ${dsets_2[j]}   --len $k --datadir ~/PA3/data/  2>&1 | tee -a $filename 
    done
fi

