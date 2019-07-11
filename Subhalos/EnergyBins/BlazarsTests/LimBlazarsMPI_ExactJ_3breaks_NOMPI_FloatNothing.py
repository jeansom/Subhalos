from __future__ import division
from __future__ import print_function

from timeit import default_timer as timer

# Tigress dirs
import os, sys
import copy

import argparse
import numpy as np
import iminuit
from iminuit import Minuit, describe, Struct
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize
# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis
import pandas as pd
import healpy as hp

# My Modules
from Recxsec_modules_NP import makeMockData, getNPTFitLL, SCDParams_Flux2Counts

ebins = 2*np.logspace(-1,3,41)[0:41]
my_iebins = [10, 15, 20, 40]

parser = argparse.ArgumentParser(description="Signal Recovery")
parser.add_argument('-x', '--xsec_inj', type=float, help='xsec to inject')
parser.add_argument('-t', '--tag', type=str, help='tag for NPTFit')
parser.add_argument('-r', '--trial', type=float, help='trial number')
parser.add_argument('-m', '--mass', type=float, help='mass to inject')
parser.add_argument('-c', '--testxsec', type=float, help='xsec to test')
args = parser.parse_args()

xsec_inj = args.xsec_inj
xsec_t = args.testxsec
Nb = 3
tag = args.tag 
mass = float(args.mass)
trial = int(args.trial)

SCD_params_arr_ebins = []
for ie in range(len(my_iebins)-1):
    SCD_params_arr_ebins.append(np.load("SCD_arrs/SCD_arr_"+str(int(mass))+"GeVExactJ_ebins_"+str(my_iebins[ie])+"-"+str(my_iebins[ie+1])+".npy"))
SCD_params_xsec_ebins = []
for ie in range(len(my_iebins)-1):
    SCD_params_xsec_ebins.append(np.load("SCD_arrs/xsec_arr_"+str(int(mass))+"GeVExactJ_ebins_"+str(my_iebins[ie])+"-"+str(my_iebins[ie+1])+".npy"))
blazar_SCD = [
    np.array([ -4.37087042 ,  2.35253149 , 1.67380836 , 1.45183683 , 39.9093125 , 0.0538183681 ]),
    np.array([ -4.83304298 ,  2.61154424 , 1.90873746 , 1.50458956 , 33.26924112 , 0.05001602 ]),
    np.array([ -5.31843419 ,  3.02252276 , 2.08354204 , 1.56432613 , 30.54236907 , 0.0400949216 ])
]

exposure_ebins= []
dif_ebins= []
iso_ebins= []
psc_ebins = []
blazars_ebins = []

for ib, b in enumerate(my_iebins[:-1]):
    fermi_exposure = np.zeros(hp.nside2npix(128))
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=1
        fermi_exposure += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/exposure'+str(bin_ind)+'.npy')
    fermi_exposure = fermi_exposure / n
    exposure_ebins.append(fermi_exposure)
    blazars_ebins.append(np.load("/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/blazarMC/blazar_map_test_"+str(b)+"_"+str(my_iebins[ib+1])+"_"+str(trial)+".npy")*fermi_exposure)

channel = 'b'
dNdLogx_df = pd.read_csv('/tigress/somalwar/Subhaloes/Subhalos/Data/AtProduction_gammas.dat', delim_whitespace=True)
dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(mass)))))[['Log[10,x]',channel]]
Egamma = np.array(mass*(10**dNdLogx_ann_df['Log[10,x]']))
dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
dNdE_interp = interp1d(Egamma, dNdEgamma)
PPnoxsec_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    ebins_temp = [ ebins[b], ebins[my_iebins[ib+1]] ]
    if ebins_temp[0] < mass:
        if ebins_temp[1] < mass:
            # Whole bin is inside
            PPnoxsec_ebins.append(1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], ebins_temp[1])[0])
        else:
            # Bin only partially contained
            PPnoxsec_ebins.append(1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], mass)[0])
    else: PPnoxsec_ebins.append(0)
PPnoxsec0_ebins = PPnoxsec_ebins.copy()

xsec0 = 1e-22
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')
subhalos = subhalos/np.mean(subhalos)

pscmask = np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)

data_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    useSubhalo = "/tigress/somalwar/Subhaloes/Subhalos/MC/FixedSCD/subhalo_flux_map_ExactJ_Einasto_"+str(int(mass))+"GeV_"+str(trial)+"_"
    data_ebins.append((np.load(useSubhalo+str(b)+"-"+str(my_iebins[ib+1])+"_1e-22.npy")*np.mean(exposure_ebins[ib])*xsec_inj/xsec0).astype(np.int32))
    data_ebins[-1][data_ebins[-1] > 1000] = 0

tag = tag + "_" + str(xsec_t)+"_"
bkg_arr = []
bkg_arr_np = []
for ib in range(len(my_iebins)-1):
    bkg_arr.append([])
    bkg_arr_np.append([[]])

ll_ebins_xsec = []
A_ebins_xsec = []
Fb_ebins_xsec = []
n_ebins_xsec = []
for ib, b in enumerate(my_iebins[:-1]):
    ll_ebins = []
    A_ebins = []
    Fb_ebins = []
    n_ebins = []
    for ix in range(len(SCD_params_arr_ebins[ib])):
        if ix == np.argmin(np.abs(SCD_params_xsec_ebins[ib] - xsec_t)):
            ll, A, Fb, n = getNPTFitLL( [data_ebins[ib]], [exposure_ebins[ib]], mask, Nb, tag, [bkg_arr[ib]], [bkg_arr_np[ib]], subhalos, False, False, True, *SCD_params_arr_ebins[ib][ix] )
            ll_ebins.append(ll[0])
            A_ebins.append(A[0])
            Fb_ebins.append(Fb[0])
            n_ebins.append(n[0])
    ll_ebins_xsec.append(np.array(ll_ebins))
    A_ebins_xsec.append(np.array(A_ebins))
    Fb_ebins_xsec.append(np.array(Fb_ebins))
    n_ebins_xsec.append(np.array(n_ebins))

ll_arr = []
SCDb_arr_ebins = []
def ll_func( xsec_t, ix, Ab_sig, Fb1_sig, Fb2_sig, Fb3_sig ): 
    return -ll_ebins_xsec[ib][ix]([ Ab_sig, Fb1_sig, Fb2_sig, Fb3_sig ])

ll = 0
for ib in range(len(my_iebins)-1):
    SCDb_arr = []
    if PPnoxsec_ebins[ib] != 0:
        print(ib, len(SCD_params_xsec_ebins))
        ix = np.argmin(np.abs(SCD_params_xsec_ebins[ib] - xsec_t))
        
        ## FLOATING NORM, MID SLOPE
        
        Fb1 = (np.array(Fb_ebins_xsec[ib][0])*((xsec_t/SCD_params_xsec_ebins[ib][ix])))[0]
        Fb2 = (np.array(Fb_ebins_xsec[ib][0])*((xsec_t/SCD_params_xsec_ebins[ib][ix])))[1]
        Fb3 = (np.array(Fb_ebins_xsec[ib][0])*((xsec_t/SCD_params_xsec_ebins[ib][ix])))[2]
        print(data_ebins[0])
        ll += -ll_func( xsec_t, 0, A_ebins_xsec[ib][0]/((xsec_t/SCD_params_xsec_ebins[ib][0])), Fb1, Fb2, Fb3 )
        #SCDb_arr.append(np.array(scipy_min.x))
        #SCDb_arr_ebins.append(SCDb_arr)
    ll_arr.append(ll)
    print( xsec_t, ll )
ll_arr = np.array(ll_arr)
print(repr(ll_arr))
