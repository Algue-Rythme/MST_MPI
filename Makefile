all: mst

mst: mst-skeleton.c mst-solution.c
	mpicc -O3 mst-skeleton.c -lm -o mst

clean:
	rm mst

smpicc:
	smpicc -O3 mst-skeleton.c -lm -o mst

smpirun:
	smpirun -np $(np) -platform platforms/clique-$(np)-platform.xml -hostfile platforms/clique-$(np)-hostfile.txt mst $(graph) $(algo)
