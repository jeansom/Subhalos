import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 05:00:00
#SBATCH --mem=5GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
trials = 100

for t in range(trials):
    batch += "#SBATCH --output=slurm/slurm_blazSCD_"+str(t)+".out\n"
    batchn = copy.copy(batch)
    batchn += "python BlazarSim.py " + str(t)
    
    fname = "batch/batchdata"+str(t)+".batch"
    f=open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname);
    os.system("sbatch " + fname);
