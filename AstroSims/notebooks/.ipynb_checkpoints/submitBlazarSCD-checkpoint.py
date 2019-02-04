import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 23:00:00
#SBATCH --mem=10GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
#SBATCH --output=slurm_BlazarSCD.out
python2 BlazarSCD.py
'''
fname = "BlazarSCD.batch"
f=open(fname, "w")
f.write(batch)
f.close()
os.system("chmod +x " + fname);
os.system("sbatch " + fname);
