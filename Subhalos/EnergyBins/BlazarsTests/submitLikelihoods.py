import sys, os
import numpy as np
import copy 

batch = '''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH -t 23:00:00
#SBATCH --mem=100GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=somalwar@princeton.edu
'''

args_arr = [ [ True, True, 1e-22, 10, 15 ],
             [ False, True, 1e-22, 10, 15 ],
#             [ True, True, 1e-24, 15, 20 ],
#             [ False, True, 1e-24, 15, 20 ],
#             [ True, True, 1e-23, 15, 20 ],
#             [ False, True, 1e-23, 15, 20 ],

#             [ True, False, 5e-24, 10, 15 ],
#             [ False, False, 5e-24, 10, 15 ],
#             [ True, False, 1e-24, 15, 20 ],
#             [ False, False, 1e-24, 15, 20 ],
#             [ True, False, 1e-23, 15, 20 ],
#             [ False, False, 1e-23, 15, 20 ]

#    [ True, True, 1e-25, 10, 12 ],
#    [ False, True, 1e-25, 10, 12 ],
#    [ True, True, 1e-25, 10, 12 ],
#    [ False, True, 1e-25, 10, 12 ],
#    [ True, True, 1e-25, 12, 15 ],
#    [ False, True, 1e-25, 12, 15 ],
#    [ True, True, 1e-25, 12, 15 ],
#    [ False, True, 1e-25, 12, 15 ]
         ]

             
ni = 0
for floatsig, minuit, xsec_inj, ebin1, ebin2 in args_arr:
    batchn = copy.copy(batch)
    tag = str(floatsig) + "_" + str(minuit) + "_" + str(xsec_inj) + "_" + str(ebin1) + "_" + str(ebin2)
    batchn += "#SBATCH --output=slurm/slurm"+str(tag)+".out\n"
    batchn += "python ResolvingPower.py " + str(floatsig) + " " + str(minuit) + " " + str(ebin1) + " " + str(ebin2) + " " + str(xsec_inj)
    
    fname = "batch/batchdata"+str(ni)+".batch"
    f=open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname);
    os.system("sbatch " + fname);
    ni+=1
