import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00
#SBATCH --mem=10GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
batch += "#SBATCH --output=slurm/slurm_siginj.out\n"
trials = 20
xsec_arr = np.logspace(-30, -20, 20)

for ix, xsec in enumerate(xsec_arr):
    for t in range(0, trials):
        batchn = copy.copy(batch)
        tag = 'siginj_check'+str(-np.log10(xsec))+'_'+str(t)+"_siginj_"
        batchn += "python LimCheck.py -n 1 -x " + str(xsec) + " -c 100 -p -5.3736545 10.0 1.32686 3.56112882 -t " + tag + " -m 100 -u False -s MC/subhalo_flux_map"+str(t)+".npy \n"

        fname = "batch/batchdata"+str(ix)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
