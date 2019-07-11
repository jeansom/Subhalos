import sys, os
import numpy as np
import copy 

trials = 1
mass = 10
if mass <= 50:
    xsec_arr = np.logspace(-25, -16, 20)
elif mass <= 500:
    xsec_arr = np.logspace(-23, -15, 20)
elif mass > 500 and mass <= 1500:
    xsec_arr = np.logspace(-22, -14, 20)
elif mass > 1500:
    xsec_arr = np.logspace(-20, -14, 20)

for t in range(trials):
    for ix, xsec in enumerate(xsec_arr):
            tag = '_FIsoFDifBlazarsFBlaz_'+str(mass)+'GeV_'+str((xsec))+"_"+str(t)+"_siginj_"
            os.system("python MakeFakeData_ExactJ_3breaks_NOMPI.py -x " + str(xsec) + " -t " + tag + " -u False -s /tigress/somalwar/Subhaloes/Subhalos/MC/FixedSCD/subhalo_flux_map_ExactJ_Einasto_"+str(mass)+"GeV_"+str(t)+"_ -r " + str(t) + " -m "+str(mass))
