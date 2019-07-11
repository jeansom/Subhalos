import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH -t 23:00:00
#SBATCH --mem=40GB
##SBATCH --ntasks-per-node=5
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''

args_arr = [ #[ True, False, True, 5e-24, 10, 15 ],
             #[ False, False, True, 5e-24, 10, 15 ],
#            # [ False, True, True, 5e-24, 10, 15 ],
             #[ True, False, True, 1e-25, 10, 15 ],
             #[ False, False, True, 1e-25, 10, 15 ],
#            # [ False, True, True, 1e-25, 10, 15 ],
             #[ True, False, True, 1e-24, 10, 15 ],
             #[ False, False, True, 1e-24, 10, 15 ],
#            # [ False, True, True, 1e-24, 10, 15 ],
             #[ True, False,  True, 1e-23, 10, 15 ],
             #[ False, False,  True, 1e-23, 10, 15 ],
#            # [ False, True,  True, 1e-23, 10, 15 ],
             #[ True, False, True, 1e-22, 10, 15 ],
             #[ False, False, True, 1e-22, 10, 15 ],
#            # [ False, True, True, 1e-22, 10, 15 ],
             #[ True, False, True, 1e-24, 15, 20 ],
             #[ False, False, True, 1e-24, 15, 20 ],
#            # [ False, True, True, 1e-24, 15, 20 ],
             #[ True, False,  True, 1e-23, 15, 20 ],
             #[ False, False,  True, 1e-23, 15, 20 ],
#            # [ False, True,  True, 1e-23, 15, 20 ],
             #[ True, False, True, 1e-22, 15, 20 ],
             #[ False, False, True, 1e-22, 15, 20 ],
#            # [ False, True, True, 1e-22, 15, 20 ],


             [ True, False, False, 1e-25, 10, 15 ],
             [ False, False, False, 1e-25, 10, 15 ],
             [ True, False, False, 1e-24, 10, 15 ],
             [ False, False, False, 1e-24, 10, 15 ],
             [ True, False, False, 5e-24, 10, 15 ],
             [ False, False, False, 5e-24, 10, 15 ],
         ]


args_arr = []
xsec_inj = np.logspace(-24, -22, 10)
for xsec in xsec_inj:
    args_arr.append( [ True, False, True, xsec, 10, 15 ] )
    args_arr.append( [ False, False, True, xsec, 10, 15 ] )
#for xsec in xsec_inj:
#    args_arr.append( [ True, False, False, xsec, 10, 15 ] )
#    args_arr.append( [ False, False, False, xsec, 10, 15 ] )
             
ni = 0
for trial in range(10):
    for floatsig, poiss, minuit, xsec_inj, ebin1, ebin2 in args_arr:
        batchn = copy.copy(batch)
        tag = "NFW"+str(floatsig) + "_" + str(poiss) + "_" + str(minuit) + "_" + str(xsec_inj) + "_" + str(ebin1) + "_" + str(ebin2) + "_" + str(trial)
        batchn += "#SBATCH --output=slurm/slurm"+str(tag)+".out\n"
        batchn += "python ResolvingPower.py " + str(floatsig) + " " + str(poiss) + " " + str(minuit) + " " + str(ebin1) + " " + str(ebin2) + " " + str(xsec_inj) + " " + str(trial)
        
        fname = "batch/batchdata"+str(tag)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname);
        os.system("sbatch " + fname);
        ni+=1
