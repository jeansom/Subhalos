import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
trials = 1

for t in range(4, 4+trials):
    for mass in [ 10, 100, 1000 ]:
        if mass <= 50:
            xsec_arr = np.logspace(-25, -16, 20)
            xsec_test_arr = np.concatenate((np.logspace(-26, -18, 25), np.logspace(-18, -15, 5)))
        elif mass <= 500:
            xsec_arr = np.logspace(-23, -15, 20)
            xsec_test_arr = np.concatenate(( np.logspace(-25, -23, 3), np.logspace( -23, -15, 25 ) ))
        elif mass > 500 and mass <= 1500:
            xsec_arr = np.logspace(-22, -14, 20)
            xsec_test_arr = np.concatenate((np.logspace(-24, -22.5, 4), np.logspace( -22, -14, 25  )))
        elif mass > 1500:
            xsec_arr = np.logspace(-20, -14, 20)
            xsec_test_arr = np.concatenate((np.logspace(-22, -20.5, 4), np.logspace( -20, -12, 25  )))
        for ix, xsec in enumerate(xsec_arr):
            for tx, xsectest in enumerate(xsec_test_arr):
                batchn = copy.copy(batch)
                batchn += "#SBATCH --mem 10GB\n"
                tag = '_FIsoFDifBlazarsFBlaz_'+str(mass)+'GeV_'+str((xsec))+'_'+str(t)+"_siginj_"
                batchn += "#SBATCH --output=slurm/slurm"+str(tag)+"_"+str(xsectest)+".out\n"
                batchn += "python LimBlazarsMPI_ExactJ_3breaks_NOMPI.py -x " + str(xsec) + " -t " + tag +" -r " + str(t) + " -m "+str(mass)+" -c " + str(xsectest) + " \n"
                
                fname = "batch/batchdata"+str(ix)+".batch"
                f=open(fname, "w")
                f.write(batchn)
                f.close()
                os.system("chmod +x " + fname);
                os.system("sbatch " + fname);
