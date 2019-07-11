from __future__ import division
from __future__ import print_function


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
from mpi4py import MPI
# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis
import pandas as pd
import healpy as hp

# My Modules
from Recxsec_modules_NP import makeMockData, getNPTFitLL, SCDParams_Flux2Counts

comm = MPI.COMM_WORLD

ebins = 2*np.logspace(-1,3,41)[0:41]
my_iebins = [10, 15]

parser = argparse.ArgumentParser(description="Signal Recovery")
parser.add_argument('-x', '--xsec_inj', type=float, help='xsec to inject')
parser.add_argument('-t', '--tag', type=str, help='tag for NPTFit')
parser.add_argument('-u', '--useMC', type=str, help='MC to use')
parser.add_argument('-s', '--useSubhalo', type=str, help='Subhalo MC to use')
parser.add_argument('-r', '--trial', type=float, help='trial number')
args = parser.parse_args()

xsec_inj = args.xsec_inj
Nb = 2
SCD_params = np.array([-5.145088814999999, 2.07503614, 1.56190097, 0.924471755, 63.699129415, 0.736])
tag = args.tag 
mass = 100
mass_inj = 100
PPnoxsec0_ebins = np.array([0.0000238640102822424, 0.00000592280390900841])
trial = int(args.trial)
if "d" in args.useMC: useMadeMC = args.useMC
else: useMadeMC = None
if "sub" in args.useSubhalo: useSubhalo = args.useSubhalo
else: useSubhalo = None

exposure_ebins= []
dif_ebins= []
iso_ebins= []
psc_ebins = []
fermi_data_ebins = []
blazars_ebins = []

for ib, b in enumerate(my_iebins[:-1]):
    fermi_exposure = np.zeros(hp.nside2npix(128))
    blazars_ebins.append(np.load("/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/blazarMC/blazar_map"+str(b)+"_"+str(my_iebins[ib+1])+"_"+str(trial)+".npy"))
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=1
        fermi_exposure += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/exposure'+str(bin_ind)+'.npy')
    fermi_exposure = fermi_exposure / n
    exposure_ebins.append(fermi_exposure)
    blazars_ebins.append(np.load("/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/blazarMC/blazar_map"+str(b)+"_"+str(my_iebins[ib+1])+"_"+str(trial)+".npy")*fermi_exposure)

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

xsec0 = 1e-22
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')

subhalo_MC = []
if useSubhalo == None: 
    for ib, b in enumerate(my_iebins[:-1]):
        fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map0_"+str(b)+"-"+str(my_iebins[ib+1])+".npy")*exposure_ebins[ib]*xsec_inj/xsec0
        fake_data = np.round(fake_data).astype(np.int32)
        subhalo_MC.append(fake_data)
else: 
    for ib, b in enumerate(my_iebins[:-1]):
        subhalo_MC.append(np.load(useSubhalo+str(b)+"-"+str(my_iebins[ib+1])+".npy")*exposure_ebins[ib]*xsec_inj/xsec0)

pscmask = np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)

data_ebins = []
if comm.rank == 0:
    for ib, b in enumerate(my_iebins[:-1]):
        #data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib]*0 ))
        data_ebins.append(np.round(np.random.poisson(subhalo_MC[ib])).astype(np.int32))
        np.save("MPITemp/fake_data"+str(ib)+"_"+str(trial), data_ebins[-1])
comm.Barrier()
if comm.rank != 0:
    for ib, b in enumerate(my_iebins[:-1]):
        data_ebins.append(np.load("MPITemp/fake_data"+str(ib)+"_"+str(trial)+".npy"))

bkg_arr = [[], []]
bkg_arr_np = [[], []]
bkg_arr_np = [[[np.ones(len(blazars_ebins[ib])), 'blaz']], [[np.ones(len(blazars_ebins[ib])), 'blaz']]]
#bkg_arr = [[[np.ones(len(blazars_ebins[ib])), 'blaz']], [[np.ones(len(blazars_ebins[ib])), 'blaz']]]

ll_ebins, A_ebins, Fb_ebins, n_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, bkg_arr_np, subhalos, False, False, True, *SCD_params )
ll_arr = []
SCDb_arr_ebins = []
def ll_func( xsec_t, Ab, n1b, n2b, n3b, Fb1b, Fb2b ): 
    return -ll_ebins[ib]([Ab, n1b, n2b, n3b, Fb1b, Fb2b, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ])
    #return -ll_ebins[ib]([A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ])

xsec_test_arr = np.logspace(-28, -20, 101)
N = len(xsec_test_arr)
my_N = np.ones(comm.size) * int(N/comm.size)
my_N[:N%comm.size] += 1
my_N = (my_N).astype(np.int32)

my_xsec_test_arr = xsec_test_arr[np.sum(my_N[:comm.rank]):np.sum(my_N[:comm.rank+1])]

print(comm.rank, my_xsec_test_arr)
for xsec_t in my_xsec_test_arr:
    ll = 0
    for ib in range(len(my_iebins)-1):
        SCDb_arr = []
        if PPnoxsec_ebins[ib] != 0:
            minuit_min = iminuit.Minuit(lambda Ab, n1b, n2b, n3b, Fb1b, Fb2b: ll_func( xsec_t, Ab, n1b, n2b, n3b, Fb1b, Fb2b ), 
                                        fix_Ab=True, fix_n1b=True, fix_n2b=True, fix_Fb1b=True, fix_Fb2b=True, 
                                        Ab=0., limit_Ab=(-10, -1), error_Ab=1e-2, n1b=10, limit_n1b=(2.05,10.), error_n1b=1e-2, n2b=0, limit_n2b=(-3,3), error_n2b=1e-2, n3b=1.4, limit_n3b=(-10, .95), error_n3b=1e-2, Fb1b=60, limit_Fb1b=(0.1,100), error_Fb1b=1e-2, Fb2b=0.5, limit_Fb2b=(0.1,1), error_Fb2b=1e-3,
                                        print_level=1)
            minuit_min.migrad()
            ll += (-minuit_min.fval)
        SCDb_arr.append([ minuit_min.values['Ab'], minuit_min.values['n1b'], minuit_min.values['n2b'], minuit_min.values['n3b'], minuit_min.values['Fb1b'], minuit_min.values['Fb2b'] ])
    SCDb_arr_ebins.append(SCDb_arr)

    ll_arr.append(ll)
ll_arr = np.array(ll_arr)
np.save("MPITemp/ll_"+str(comm.rank)+"_"+str(trial), np.array(ll_arr))
np.save("MPITemp/SCDb_"+str(comm.rank)+"_"+str(trial), np.array(SCDb_arr_ebins))
comm.Barrier()

if comm.rank == 0:
    ll_arr = np.empty(len(xsec_test_arr))
    SCD_arr_ebins = []
    for i in range(comm.size):
        ll_arr[np.sum(my_N[:i]):np.sum(my_N[:i+1])] = np.load("MPITemp/ll_"+str(i)+"_"+str(trial)+".npy")
        SCD_arr_ebins.append(np.load("MPITemp/SCDb_"+str(i)+"_"+str(trial)+".npy"))
    TS_xsec_ary = 2*(ll_arr - ll_arr[0])
    max_loc = np.argmax(TS_xsec_ary)
    max_TS = TS_xsec_ary[max_loc]
    
    xsec_rec = 1e-50
    for xi in range(max_loc, len(xsec_test_arr)):
        val = TS_xsec_ary[xi] - max_TS
        if val < -2.71:
            scale = (TS_xsec_ary[xi-1]-max_TS+2.71)/(TS_xsec_ary[xi-1]-TS_xsec_ary[xi])
            xsec_rec = xsec_test_arr[xi-1] + scale*(xsec_test_arr[xi] - xsec_test_arr[xi-1])
            break
    np.savez("lim_"+str(xsec_inj) + "_" + tag, xsec_rec, ll_arr, SCDb_arr_ebins )
    print("Recovered: ", xsec_rec)
