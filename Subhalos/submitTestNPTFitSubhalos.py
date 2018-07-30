import sys, os
import numpy as np

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH --mem=10GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
batch += "#SBATCH --output=slurm/slurmNPTFit.out\n"
batch += "python TestNPTFitSubhalos.py \n"
fname = "batch/batch.batch"
f=open(fname, "w")
f.write(batch)
f.close()
os.system("chmod +x " + fname);
os.system("sbatch " + fname);
