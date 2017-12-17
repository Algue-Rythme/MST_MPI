#!/bin/bash
d=$1
algo=$2
echo "Print dense graphs from folder dense with algorithm $algo"
for j in {1000..9000..1000}
do
    nbedges=$(expr $j \* $(expr $j - 1) / 2 / $d)
    for i in {1..10}
    do
        echo "train on dense$d/graph$j\_$nbedges\_$nbedges\.txt $algo-seq : $i"
        mpirun -np 1 ./mst dense$d/graph$j\_$nbedges\_$nbedges\.txt $algo-seq > out.txt 2>> dense$d$algo.txt
    done
done
