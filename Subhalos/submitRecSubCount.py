import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 0:59:00
#SBATCH --mem=10GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
batch += "#SBATCH --output=slurm/slurm.out\n"
trials = 20
#mass_arr = np.array([1.00000000e+01,1.50000000e+01,2.00000000e+01,2.50000000e+01,3.00000000e+01,4.00000000e+01,5.00000000e+01,6.00000000e+01,7.00000000e+01,8.00000000e+01,9.00000000e+01,1.00000000e+02,1.10000000e+02,1.20000000e+02,1.30000000e+02,1.40000000e+02,1.50000000e+02,1.60000000e+02,1.80000000e+02,2.00000000e+02,2.20000000e+02,2.40000000e+02,2.60000000e+02,2.80000000e+02,3.00000000e+02,3.30000000e+02,3.60000000e+02,4.00000000e+02,4.50000000e+02,5.00000000e+02,5.50000000e+02,6.00000000e+02,6.50000000e+02,7.00000000e+02,7.50000000e+02,8.00000000e+02,9.00000000e+02,1.00000000e+03,1.10000000e+03,1.20000000e+03,1.30000000e+03,1.50000000e+03,1.70000000e+03,2.00000000e+03,2.50000000e+03,3.00000000e+03,4.00000000e+03,5.00000000e+03,6.00000000e+03,7.00000000e+03,8.00000000e+03,9.00000000e+03,1.00000000e+04])
mass_arr = np.array([10.0])

for im, mass in enumerate(mass_arr):
    for t in range(trials):
        batchn = copy.copy(batch)
        tag = 'subs'+str(mass)+'_'+str(t)
        #batchn += "python Lim.py -n 1 -x 2.310129700083158e-24 -c 100 -p 3.53399 10.0 1.89914 -7.71429 -t " + tag + " -m "+str(mass) + " -u MC/MCBkg/mockdata_subs10.0"+'_'+str(t) + ".npy -s False \n"
        batchn += "python Lim.py -n 1 -x 2.310129700083158e-24 -c 100 -p 3.53399 10.0 1.89914 -7.71429 -t " + tag + " -m "+str(mass) + " -u False -s MC/subhalo_flux_map"+str(t)+".npy \n"
        fname = "batch/batchdata"+str(im)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
