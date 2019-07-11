#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

print("-"*78)
print(" Running on %d cores" % comm.size)
print("-"*78)

N = 11
my_N = np.ones(comm.size) * int(N/comm.size)
my_N[:N%comm.size] += 1
my_N = (my_N).astype(np.int32)

A = np.arange(N, dtype=np.float64)
my_A = A[np.sum(my_N[:comm.rank]):np.sum(my_N[:comm.rank+1])]

print("After Scatter:")
for r in range(comm.size):
    if comm.rank == r:
        print("[%d] %s" % (comm.rank, my_A))
    comm.Barrier()

# Everybody is multiplying by 2
my_A *= 2
np.save("my_A"+str(comm.rank), my_A)
A[np.sum(my_N[:comm.rank]):np.sum(my_N[:comm.rank+1])] = my_A

comm.Barrier()

if comm.rank == 0:
    for i in range(comm.size):
        A[np.sum(my_N[:i]):np.sum(my_N[:i+1])] = np.load("my_A"+str(i)+".npy")
    print("After Allgather:", A)
