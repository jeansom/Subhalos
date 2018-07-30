import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=5
#SBATCH -t 23:00:00
#SBATCH --mem=10GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
trials = 5
xsec_arr = np.logspace(-30, -20, 30)
#xsec_arr = np.array([0.0, 1e-23])

for ix, xsec in enumerate(xsec_arr):
    for t in range(trials):
        batchn = copy.copy(batch)
        tag = 'siginj_blazarstemp_'+str((xsec))+'_'+str(t)+"_siginj_"
        batchn += "#SBATCH --output=slurm/slurm"+str(tag)+".out\n"
        batchn += "mpiexec -n 5 python LimBlazarsTemp.py -n 1 -x " + str(xsec) + " -c 100 -p -5.71613418 -5.52961428 10. 1.79143877 10. 1.80451023 2.99124533 2.38678777 -t " + tag + " -m 100 -u False -s /tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map"+str(t)+"_ -pp 0.0000238640102822424 0.00000592280390900841 -r " + str(t) + " \n"

        fname = "batch/batchdata"+str(ix)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
