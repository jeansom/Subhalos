import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 7:00:00
#SBATCH --mem=10GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
batch += "#SBATCH --output=slurm/slurm.out\n"
trials = 20
mass_arr = np.array([1.000000000e+01, 1.50000000e+01,2.00000000e+01,2.50000000e+01,3.00000000e+01,4.00000000e+01,5.00000000e+01,6.00000000e+01,7.00000000e+01,8.00000000e+01,9.00000000e+01,1.00000000e+02,1.10000000e+02,1.20000000e+02,1.30000000e+02,1.40000000e+02,1.50000000e+02,1.60000000e+02,1.80000000e+02,2.00000000e+02,2.20000000e+02,2.40000000e+02,2.60000000e+02,2.80000000e+02,3.00000000e+02,3.30000000e+02,3.60000000e+02,4.00000000e+02,4.50000000e+02,5.00000000e+02,5.50000000e+02,6.00000000e+02,6.50000000e+02,7.00000000e+02,7.50000000e+02,8.00000000e+02,9.00000000e+02,1.00000000e+03,1.10000000e+03,1.20000000e+03,1.30000000e+03,1.50000000e+03,1.70000000e+03,2.00000000e+03,2.50000000e+03,3.00000000e+03,4.00000000e+03,5.00000000e+03,6.00000000e+03,7.00000000e+03,8.00000000e+03,9.00000000e+03,1.00000000e+04])
#mass_arr = np.array([10.0])

for im, mass in enumerate(mass_arr):
    for t in range(trials):
        batchn = copy.copy(batch)
        tag = 'lim_5_'+str(mass)+'_'+str(t)
        #batchn += "python Lim.py -n 1 -x 0. -c 100 -p  -5.71613418 -5.52961428 10. 1.79143877 10. 1.80451023 2.99124533 2.38678777 -t " + tag + " -m " +str(mass) + " -u /tigress/somalwar/Subhaloes/Subhalos/MC/mockdata_lim_5_10.0_"+str(t)+".npy -s False -pp  0.0000238640102822424 0.00000592280390900841 \n"
        batchn += "python Lim.py -n 1 -x 0. -c 100 -p -5.71613418 -5.52961428 10. 1.79143877 10. 1.80451023 2.99124533 2.38678777 -t " + tag + " -m " +str(mass) + " -u False -s /tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map"+str(t)+"_ -pp  0.0000238640102822424 0.00000592280390900841 \n"

        fname = "batch/batchdata"+str(im)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
