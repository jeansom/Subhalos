import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=10
#SBATCH -t 23:00:00
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
trials = 5
xsec_arr = np.logspace(-23, -15, 20)

for t in range(trials):
    for ix, xsec in enumerate(xsec_arr):
        batchn = copy.copy(batch)
        batchn += "#SBATCH --mem 40GB\n"
        tag = '_FIsoFDifBlazarsFBlaz_100GeV_'+str((xsec))+'_'+str(t)+"_siginj_"
        batchn += "#SBATCH --output=slurm/slurm"+str(tag)+".out\n"
        batchn += "mpiexec -n 10 python LimBlazarsMPI_ExactJ_3breaks.py -x " + str(xsec) + " -t " + tag + " -u False -s /tigress/somalwar/Subhaloes/Subhalos/MC/FixedSCD/subhalo_flux_map_ExactJ_Einasto_100GeV_"+str(t)+"_ -r " + str(t) + " -m 100 \n"

        fname = "batch/batchdata"+str(ix)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
