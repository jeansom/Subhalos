import sys, os
import numpy as np
import copy

trials = 5
xt_a = [1e-22]
xi_a = [1e-22]

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
    for xi_t in xi_a:
        for xt_t in xt_a:
            batchn = copy.copy(batch)
            batchn += "#SBATCH --output=slurm/slurm.out\n"
            batchn += " python /tigress/somalwar/SubhalosFresh/Code/nbArgs.py RunLimit.ipynb \"{'xsec_inj': "+str(xi_t)+", 'tag': 'test', 'trial': "+str(t)+", 'mass': 100, 'xsec_t': "+str(xt_t)+"}\" "
            
            fname = "batch/batchdata.batch"
            f=open(fname, "w")
            f.write(batchn)
            f.close()
            os.system("chmod +x " + fname);
            os.system("sbatch " + fname);
