import sys, os
import numpy as np
import copy

trials = 5

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 05:00:00
#SBATCH --mem=4GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
for t in range(trials):
    batchn = copy.copy(batch)
    batchn += "#SBATCH --output=slurm/slurm.out\n"
    batchn += " python /tigress/somalwar/SubhalosFresh/Code/nbArgs.py FindSubhaloSCD.ipynb \"{'xsec': 1e-22, 'trial': "+str(t)+", 'mass': 100}\" "
    
    fname = "batch/batchdata.batch"
    f=open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname);
    os.system("sbatch " + fname);
