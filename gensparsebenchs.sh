#!/bin/bash
d=$1
algo=$2
echo "Print sparse graphs from folder sparse with algorithm $algo"
for j in {1..10}
do
    nbedges=$(expr $j \* $d)
    for i in {1..10}
    do
        echo "train on sparse$d/graph$j\000\_$nbedges\000\_$nbedges\000.txt $algo-seq : $i"
        mpirun -np 1 ./mst sparse$d/graph$j\000\_$nbedges\000\_$nbedges\000.txt $algo-seq > out.txt 2>> sparse$d$algo.txt
    done
done
