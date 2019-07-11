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
my_iebins = [10, 17, 24, 27]

parser = argparse.ArgumentParser(description="Signal Recovery")
parser.add_argument('-x', '--xsec_inj', type=float, help='xsec to inject')
parser.add_argument('-t', '--tag', type=str, help='tag for NPTFit')
parser.add_argument('-u', '--useMC', type=str, help='MC to use')
parser.add_argument('-s', '--useSubhalo', type=str, help='Subhalo MC to use')
parser.add_argument('-r', '--trial', type=float, help='trial number')
args = parser.parse_args()

xsec_inj = args.xsec_inj
Nb = 2
#SCD_params = np.array([-5.43320258, 2.05, 1.76690669, 1.29602111, 60.79630843, 0.1])
SCD_params = np.array([-5.44404671, 2.05, 1.76208101, 1.29979701, 61.41696489, 0.1])
tag = args.tag 
mass = 100
mass_inj = 100
PPnoxsec0_ebins = np.array([   ])
my_iebins0 = np.array([ int(sys.argv[1]), int(sys.argv[2]) ])
print(my_iebins0)
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

def PPnoxsec( ebin_low, ebin_high ):
    ebins_temp = [ ebin_low, ebin_high ]
    if ebins_temp[0] < mass:
        if ebins_temp[1] < mass:
            # Whole bin is inside
            return 1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], ebins_temp[1])[0]
        else:
            # Bin only partially contained
            return 1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], mass)[0]
    else: return 0

for ib, b in enumerate(my_iebins[:-1]):
    PPnoxsec_ebins.append(PPnoxsec( ebins[b], ebins[my_iebins[ib+1]] ))

PPnoxsec0 = PPnoxsec( ebins[my_iebins0[0]], ebins[my_iebins0[1]] ))

xsec0 = 1e-22
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')
subhalos = subhalos/np.mean(subhalos)

subhalo_MC = []
if useSubhalo == None: 
    for ib, b in enumerate(my_iebins[:-1]):
        fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map0_"+str(my_iebins0[0])+"-"+str(my_iebins0[1])+".npy")*exposure_ebins[ib]*xsec_inj/xsec0*PPnoxsec[ib]/PPnoxsec0
        fake_data = np.round(fake_data).astype(np.int32)
        subhalo_MC.append(fake_data)
else: 
    for ib, b in enumerate(my_iebins[:-1]):
        subhalo_MC.append(np.round(np.load(useSubhalo+str(my_iebins0[0])+"-"+str(my_iebins0[1])+".npy")*exposure_ebins[ib]*xsec_inj/xsec0*PPnoxsec[ib]/PPnoxsec0).astype(np.int32))

pscmask = np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)

data_ebins = []
if comm.rank == 0:
    for ib, b in enumerate(my_iebins[:-1]):
        #data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib]*0 ))
        data_ebins.append(np.round(np.random.poisson(subhalo_MC[ib])).astype(np.int32))
        np.save("MPITemp/fake_data"+str(ib)+"_"+tag, data_ebins[-1])
comm.Barrier()
if comm.rank != 0:
    for ib, b in enumerate(my_iebins[:-1]):
        data_ebins.append(np.load("MPITemp/fake_data"+str(ib)+"_"+tag+".npy"))

bkg_arr = [[], []]
#bkg_arr_np = [[], []]
bkg_arr_np = [[[np.ones(len(blazars_ebins[ib])), 'blaz']], [[np.ones(len(blazars_ebins[ib])), 'blaz']]]

ll_ebins, A_ebins, Fb_ebins, n_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, bkg_arr_np, subhalos, False, False, True, *SCD_params )
ll_arr = []
SCDb_arr_ebins = []

blazar_spectrum = np.load("/tigress/somalwar/Subhaloes/Subhalos/blazars/blazar_spectrum.npz")['arr_0']

blazar_scale = []

for ib in range(len(my_iebins)-1):
    blazar_scale.append( np.trapz(blazar_spectrum[my_iebins[ib], my_iebins[ib+1]], my_ebins[my_iebins[ib], my_iebins[ib+1]]) )
blazar_scale = np.array(blazar_scale)
blazar_scale[1:] = blazar_scale[1:]/blazar_scale[0]

def ll_func( xsec_t, Ab, n1b, n2b, n3b, Fb1b, Fb2b ): 
    ll = 0
    for ib in range(len(my_iebins)-1):
        ll += -ll_ebins[ib]([Ab-np.log10(blazar_scale[ib]), n1b, n2b, n3b, Fb1b*blazar_scale[ib], Fb2b, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0)) ])
    return ll
#    return -ll_ebins[ib]([A_ebins[ib]/((xsec_t/xsec0)), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0))) ])

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

            scipy_min = minimize( lambda args: ll_func( xsec_t, *args ),
                                  [-6, 5, 0, 0.1, 60, 0.5], method="SLSQP", bounds = [ [-10, -2], [2.05,10], [-3,3], [-10, 1.95], [1,100], [0.1, 1.] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            ll += -scipy_min.fun
            SCDb_arr.append(np.array( [scipy_min.x[0], scipy_min.x[1], scipy_min.x[2], scipy_min.x[3], scipy_min.x[4], scipy_min.x[5] ]))

            '''
            ll += -ll_func( xsec_t, 0., 1, 1., 0.1, 1000000., 0.5 )
            SCDb_arr.append([0., 5, 0, 0.1, 60, 0.5])
            '''
            SCDb_arr_ebins.append(SCDb_arr)
    ll_arr.append(ll)
    print( xsec_t, ll )
ll_arr = np.array(ll_arr)
np.save("MPITemp/ll_"+str(comm.rank)+"_"+tag, np.array(ll_arr))
np.save("MPITemp/SCDb_"+str(comm.rank)+"_"+tag, np.array(SCDb_arr_ebins))
comm.Barrier()

if comm.rank == 0:
    ll_arr = np.empty(len(xsec_test_arr))
    SCD_arr_ebins = []
    for i in range(comm.size):
        ll_arr[np.sum(my_N[:i]):np.sum(my_N[:i+1])] = np.load("MPITemp/ll_"+str(i)+"_"+tag+".npy")
        SCD_arr_ebins.append(np.load("MPITemp/SCDb_"+str(i)+"_"+tag+".npy"))
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
    print("Best fit: ", xsec_test_arr[max_loc])
