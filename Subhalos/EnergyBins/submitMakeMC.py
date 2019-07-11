import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 05:00:00
#SBATCH --mem=4GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
trials = 100
#xsec_test_arr = np.concatenate( ([1e-22], np.logspace(-21, -15, 101)) ) # Old
#xsec_test_arr = np.concatenate(( np.logspace(-20, -12, 25), [1e-22] )) # 10 TeV
#xsec_test_arr = np.logspace(-23, -15, 25) # 100 GeV
#xsec_test_arr = np.logspace(-25, -17, 25)[:6] # 100 GeV
#xsec_test_arr = np.logspace(-26, -18, 25) # 10 GeV
#xsec_test_arr = np.logspace( -18, -15, 5 ) # 10 GeV
xsec_test_arr = np.concatenate((np.logspace(-24, -22.5, 4), np.logspace( -22, -14, 25  )))
for t in range(trials):
    for xsec in xsec_test_arr:
        batchn = copy.copy(batch)
        batchn += "#SBATCH --output=slurm/slurm1TeVExactJ_ebins_"+str(t)+"_"+str(xsec)+".out\n"
        batchn += "python SubhalosModularEBinsScipy_ExactJ.py " + str(t) + " " + str(xsec)
        #    batchn += "python BlazarSim.py " + str(t)
        
        fname = "batch/batchdata_"+str(t)+"_"+str(xsec)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
