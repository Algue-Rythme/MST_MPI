from matplotlib import pyplot as plt
from math import *

xnodes = [i*1000 for i in range(1,11)]
def xedges(d):
    return [i*(i-1)/(2*d) for i in range(1000,10000,1000)]

def plot_from(x, sparsity, d, color):
    for algo, marker in zip(["prim", "kruskal"], ['s', 'd']):
        f = open(sparsity+str(d)+algo+".txt", "r")
        data = [float(f) for f in f.read().split()]
        means = [sum(data[i:i+10])/10. for i in range(0,len(x)*10,10)]
        plt.plot(x, means, color=color, marker=marker)

def plot_parallel_from(file1, file2):
    x = [1, 2, 4, 8, 16, 32, 64]
    for name, color in zip([file1, file2], ["red", "blue"]):
        f = open(name, "r")
        data = [float(i) for i in f.read().split()]
        plt.plot(x[:len(data)], data, color=color, marker='d')

#plot_from(xnodes, "sparse", 5, 'red')
#plot_from(xnodes, "sparse", 10, 'green')
#plot_from(xnodes, "sparse", 25, 'blue')

#plot_from(xedges(2), "dense", 2, 'red')

# plot_parallel_from("sparse25primpar.txt", "sparse25kruskalpar.txt")

plot_parallel_from("dense2primpar.txt", "dense2kruskalpar.txt")

plt.yscale('log', basey=10)
plt.xscale('log', basex=10)
plt.show()
