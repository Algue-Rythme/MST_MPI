#!/bin/bash
folder=$1
d=$2
echo "Print sparse graphs in folder $folder"
for i in {1000..10000..1000}
do
    nbedges=$(expr $i \* $d)
    python create-graph.py $i $nbedges $nbedges $folder/graph$i\_$nbedges\_$nbedges.txt
done
