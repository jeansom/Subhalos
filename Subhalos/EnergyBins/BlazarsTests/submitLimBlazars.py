import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=20
#SBATCH -t 23:00:00
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
trials = 5
#xsec_arr = np.concatenate((np.logspace(-30, -25, 15), np.logspace(-25, -15, 30)[1:] ))
xsec_arr = np.logspace(-25, -15, 30)
#xsec_arr = np.logspace(-25, -15, 30)[1:][7:22]
#xsec_arr = np.logspace(-25, -15, 30)[1:]
#xsec_arr = np.array([1e-23])

for t in range(trials):
    for ix, xsec in enumerate(xsec_arr):
        batchn = copy.copy(batch)
        if xsec in np.logspace(-25, -15, 30)[1:][7:22]: 
            batchn += "#SBATCH --mem 100GB\n"
        else: 
            batchn += "#SBATCH --mem 60GB\n"
        print(xsec)
        tag = '_FIsoFDifBlazarsFBlaz_1000GeV_'+str((xsec))+'_'+str(t)+"_siginj_"
        batchn += "#SBATCH --output=slurm/slurm"+str(tag)+".out\n"
        #batchn += "mpiexec -n 20 python LimBlazarsMPI.py -e 10 15 -x " + str(xsec) + " -t " + tag + " -u False -s /tigress/somalwar/Subhaloes/Subhalos/MC/FixedSCD/subhalo_flux_map_"+str(t)+"_ -r " + str(t) + " \n"
        batchn += "mpiexec -n 10 python LimBlazarsMPI_FloatBreaks.py -x " + str(xsec) + " -t " + tag + " -u False -s /tigress/somalwar/Subhaloes/Subhalos/MC/FixedSCD/subhalo_flux_map_"+str(t)+"_ -r " + str(t) + " \n"

        fname = "batch/batchdata"+str(ix)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
