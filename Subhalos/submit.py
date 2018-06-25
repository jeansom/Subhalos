import sys, os

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
#SBATCH --mem=10GB
#SBATCH -p hepheno
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''
trials = int(6e14/1e8)
#trials = 1
for trial in range(trials):
    batchn = batch
    batchn += "#SBATCH --output=slurm/slurm"+str(trial)+".out\n"
    batchn += "python SubmitSubhalos.py -t " + str(trial) +"\n"
    fname = "batch/batch_"+str(trial)+".batch"
    f=open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname);
    os.system("sbatch " + fname);
