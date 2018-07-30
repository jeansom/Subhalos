import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=5
#SBATCH -t 23:00:00
#SBATCH --mem=100GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
trials = 5
#xsec_arr = np.logspace(-30, -20, 30)
xsec_arr = np.array([1e-25])

for ix, xsec in enumerate(xsec_arr):
    for t in range(trials):
        batchn = copy.copy(batch)
        tag = 'siginj_flblaz_wbl_'+str((xsec))+'_'+str(t)+"_siginj_"
        batchn += "#SBATCH --output=slurm/slurm"+str(tag)+".out\n"
        batchn += "mpiexec -n 5 python LimBlazarsTemp.py -x " + str(xsec) + " -t " + tag + " -u False -s /tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map"+str(t)+"_ -r " + str(t) + " \n"

        fname = "batch/batchdata"+str(ix)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
