#!/bin/bash

#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node 4

mpiexec -n 4 mpitest.py
