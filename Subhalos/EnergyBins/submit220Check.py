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
trials = 20
xsec_arr = np.logspace(-30, -20, 10)

for ix, xsec in enumerate(xsec_arr):
    for t in range(20, 20+trials):
        batchn = copy.copy(batch)
        tag = 'siginj_subs'+str(-np.log10(xsec))+'_'+str(t)+"_220test_init_"
        batchn += "#SBATCH --output=slurm/slurm"+str(tag)+".out\n"
        batchn += "python LimCheck.py -n 1 -x " + str(xsec) + " -c 100 -p  -5.83473528 10.0 1.78793424 3.1179019 -t " + tag + " -m 100 -u False -s /tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map"+str(t)+"_ -pp  2.9786839931645153e-05 \n"

        fname = "batch/batchdata"+str(ix)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
