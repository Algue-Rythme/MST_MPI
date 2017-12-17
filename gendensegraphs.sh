#!/bin/bash
folder=$1
frac=$2
echo "Print dense graphs in folder $folder"
for i in {1000..9000..1000}
do
    nbedges=$(expr $i \* $(expr $i - 1) / 2 / $frac)
    python create-graph.py $i $nbedges $nbedges $folder/graph$i\_$nbedges\_$nbedges.txt
done
