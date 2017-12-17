#!/usr/bin/env python3

import sys
import os
import math

## https://milkyway.cs.rpi.edu/milkyway/cpu_list.php

link_latency = "10us"
link_bandwidth = "10Gbps"

def issueHead():
    head = ("<?xml version='1.0'?>\n"
            "<!DOCTYPE platform SYSTEM \"http://simgrid.gforge.inria.fr/simgrid/simgrid.dtd\">\n"
            "<platform version=\"4.1\">\n\n")
    config =    ("<config>\n"
                "<prop id=\"smpi/simulate-computation\" value=\"yes\"></prop>\n"
                "<prop id=\"smpi/host-speed\" value=\"1175000000\"></prop>\n"
                "</config>\n\n")
    zone_head = "<zone id=\"AS0\" routing=\"Dijkstra\">\n"
    return head + config + zone_head

def issueTail():
	return "</zone>\n</platform>\n"

def issueHost(index):
    return "  <host id=\"host-"+str(index)+".algue.fr\" speed=\""+sim_compute_power+"\"/>\n"

def issueLink(x):
    return "  <link id=\"link-"+str(x)+"\" latency=\""+str(link_latency)+"\" bandwidth=\""+link_bandwidth+"\"/>\n"

def issueRoute(x, y, link):
    return "  <route src=\"host-"+str(x)+".algue.fr\" dst=\"host-"+str(y)+".algue.fr\"><link_ctn id=\"link-"+str(link)+"\"/></route>\n"

if (len(sys.argv) != 5):
	print("Usage: gen_clique.py <num hosts> <simulation-node-compute-power> <simulation-link-bandwidth> <simulation-link-latency> \n")
	print("Example: gen_clique.py 32 100Gf 10Gbps 10us \n")
	print("  Will generate a clique-<num hosts>-platform.xml and clique-<num hosts>-hostfile.txt file\n")
	exit(1)

num_hosts = int(sys.argv[1])
sim_compute_power = sys.argv[2]+"Gf"
link_bandwidth = sys.argv[3]
link_latency = sys.argv[4]

filename = "./platforms/clique-"+str(num_hosts)+"-platform.xml"
fh = open(filename, 'w')
fh.write(issueHead())

for i in range(num_hosts):
    fh.write(issueHost(i+1))

for i in range(num_hosts*(num_hosts-1)//2):
    fh.write(issueLink(i+1))

numLink = 0
for i in range(num_hosts):
    for j in range(i):
        fh.write(issueRoute(j+1, i+1, numLink+1))
        numLink += 1

fh.write(issueTail())
fh.close()
print(filename+" created\n")
filename = "./platforms/clique-"+str(num_hosts)+"-hostfile.txt"
fh = open(filename, 'w')
for i in range(num_hosts):
    fh.write("host-"+str(i+1)+".algue.fr\n")
fh.close()
print(filename+" created\n")
