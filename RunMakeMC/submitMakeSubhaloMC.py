import sys, os
import numpy as np
import copy

trials = 1
x_a = [1e-22]
mass = 100

# Defining some constants
M_MW = 1.1e12 # [M_s]

N_calib = 300. # Number of subhalos with masses 10^8 - 10^10 M_sun
mMin_calib = 1e8 # [M_s]
mMax_calib = 1e10 # [M_s]

mMin = 1e-5*M_MW
mMax = .01*M_MW # [M_s]

def dNdm_func(m): # Subhalo mass function
    norm = N_calib / ( -.9**(-1) * (mMax_calib**(-.9) - mMin_calib**(-.9)))
    return norm * (m)**(-1.9)

N_subs = np.random.poisson( round(N_calib / ( -.9**(-1) * (mMax_calib**(-.9) - mMin_calib**(-.9))) * -.9**(-1) * (mMax**(-.9) - mMin**(-.9))) ) # Total number of subhalos
N_subs = 150*4 + 10
n_max = 150
n_iters = int(np.floor(N_subs/n_max))
n_ex = N_subs - n_max * n_iters

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 05:00:00
#SBATCH --mem=4GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
for x in x_a:
    for t in range(trials):
        for i in range(n_iters+1):
            nsubs = (i==n_iters)*n_ex + (i!=n_iters)*n_max
            if nsubs==0: continue
            print(nsubs)
            batchn = copy.copy(batch)
            batchn += "#SBATCH --output=slurm/slurm.out\n"
            batchn += " python /tigress/somalwar/SubhalosFresh/Code/nbArgs.py MakeSubhaloMC.ipynb \"{'xsec': "+str(x)+", 'nsubs': "+str(nsubs)+", 'trial': "+str(t)+", 'iter': "+str(i)+", 'mass': "+str(mass)+"}\" "
            
            fname = "batch/batchdata.batch"
            f=open(fname, "w")
            f.write(batchn)
            f.close()
            os.system("chmod +x " + fname);
            os.system("sbatch " + fname);
