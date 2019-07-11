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
my_iebins = [10, 15, 20, 40]

parser = argparse.ArgumentParser(description="Signal Recovery")
parser.add_argument('-x', '--xsec_inj', type=float, help='xsec to inject')
parser.add_argument('-t', '--tag', type=str, help='tag for NPTFit')
parser.add_argument('-u', '--useMC', type=str, help='MC to use')
parser.add_argument('-s', '--useSubhalo', type=str, help='Subhalo MC to use')
parser.add_argument('-r', '--trial', type=float, help='trial number')
parser.add_argument('-m', '--mass', type=float, help='mass to inject')
args = parser.parse_args()

xsec_inj = args.xsec_inj
Nb = 3

SCD_params_arr_ebins = []
for ie in range(len(my_iebins)-1):
    SCD_params_arr_ebins.append(np.load("SCD_arrs/SCD_arr_100GeVExactJ_ebins_"+str(my_iebins[ie])+"-"+str(my_iebins[ie+1])+".npy"))
SCD_params_xsec_ebins = []
for ie in range(len(my_iebins)-1):
    SCD_params_xsec_ebins.append(np.load("SCD_arrs/xsec_arr_100GeVExactJ_ebins_"+str(my_iebins[ie])+"-"+str(my_iebins[ie+1])+".npy"))
blazar_SCD = [
    np.array([ -4.37087042 ,  2.35253149 , 1.67380836 , 1.45183683 , 39.9093125 , 0.0538183681 ]),
    np.array([ -4.83304298 ,  2.61154424 , 1.90873746 , 1.50458956 , 33.26924112 , 0.05001602 ]),
    np.array([ -5.31843419 ,  3.02252276 , 2.08354204 , 1.56432613 , 30.54236907 , 0.0400949216 ])
]

np.array([-4.733327518453883, 2.4461263, 1.77293727, 1.48618555, 60.55017686, 0.1 ])
tag = args.tag 
mass = float(args.mass)
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
    dif = np.zeros(len(fermi_exposure))
    iso = np.zeros(len(fermi_exposure))
    psc = np.zeros(len(fermi_exposure))
    data = np.zeros(len(fermi_exposure))
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=1
        fermi_exposure += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/exposure'+str(bin_ind)+'.npy')
        dif += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/dif'+str(bin_ind)+'.npy')
        iso += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/iso'+str(bin_ind)+'.npy')
        psc += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/psc'+str(bin_ind)+'.npy')
        data += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/data'+str(bin_ind)+'.npy')
    fermi_exposure = fermi_exposure / n
    dif_ebins.append(dif)
    iso_ebins.append(iso)
    psc_ebins.append(psc)
    fermi_data_ebins.append(data.astype(np.int32))
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

subhalo_MC = []
if useSubhalo == None: 
    for ib, b in enumerate(my_iebins[:-1]):
        fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map_NFW_"+str(b)+"-"+str(my_iebins[ib+1])+"_"+str(SCD_params_xsec_ebins[ib][np.argmin(np.abs(SCD_params_xsec_ebins[ib] - xsec_inj))])+".npy")*np.mean(exposure_ebins[ib])*xsec_inj/SCD_params_xsec_ebins[ib][np.argmin(np.abs(SCD_params_xsec_ebins[ib] - xsec_inj))]
        fake_data = np.round(np.random.poisson(fake_data)).astype(np.int32)
        subhalo_MC.append(fake_data)
        subhalo_MC[-1][subhalo_MC[-1] > 1000] = 0
else: 
    for ib, b in enumerate(my_iebins[:-1]):
        subhalo_MC.append(np.load(useSubhalo+str(b)+"-"+str(my_iebins[ib+1])+"_1e-22.npy")*np.mean(exposure_ebins[ib])*xsec_inj/xsec0)
        subhalo_MC[-1][subhalo_MC[-1] > 1000] = 0

pscmask = np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)

data_ebins = []
if comm.rank == 0:
    for ib, b in enumerate(my_iebins[:-1]):
        n_bkg = nptfit.NPTF(tag='norm')
        n_bkg.load_data(fermi_data_ebins[ib].copy(), exposure_ebins[ib].copy())
        n_bkg.load_mask(mask)
        
        n_bkg.add_template(dif_ebins[ib].copy(), 'dif')
        n_bkg.add_template(iso_ebins[ib].copy(), 'iso')
        n_bkg.add_template(psc_ebins[ib].copy(), 'psc')
        
        n_bkg.add_poiss_model('dif', '$A_\mathrm{dif}$', [0,20], False)
        n_bkg.add_poiss_model('iso', '$A_\mathrm{iso}$', [0,20], False)
        n_bkg.add_poiss_model('psc', '$A_\mathrm{psc}$', [0,20], False)

        n_bkg.configure_for_scan()

        bkg_min = minimize( lambda args: -n_bkg.ll([*args]), 
                            [ 0.89, 5, 0.03795109 ], method="SLSQP", bounds = [ [0,10], [0,10], [0,10] ], options={'ftol':1e-15, 'eps':1e-10, 'maxiter':5000, 'disp':True} )
        print(bkg_min.x)
        data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib], bkg_min.x[0]*dif_ebins[ib], bkg_min.x[1]*iso_ebins[ib] ))
        np.save("MPITemp/fake_data"+str(ib)+"_"+tag, data_ebins[-1])
comm.Barrier()
if comm.rank != 0:
    for ib, b in enumerate(my_iebins[:-1]):
        data_ebins.append(np.load("MPITemp/fake_data"+str(ib)+"_"+tag+".npy"))

bkg_arr = []
bkg_arr_np = []
for ib in range(len(my_iebins)-1):
    bkg_arr.append([ [ dif_ebins[ib], 'dif'], [iso_ebins[ib], 'iso']] )
    bkg_arr_np.append([[np.ones(len(blazars_ebins[ib])), 'blaz']])

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
def ll_func( xsec_t, ix, A_dif, A_iso, A_psc, Ab, n1b, n2b, n3b, Fb1b, Fb2b, Ab_sig, Fb1_sig, Fb2_sig, Fb3_sig ): 
    return -ll_ebins_xsec[ib][ix]([ A_dif, A_iso, Ab, n1b, n2b, n3b, Fb1b, Fb2b, Ab_sig, Fb1_sig, Fb2_sig, Fb3_sig ])

xsec_test_arr = np.logspace( -30, -15, 101 )
if mass <= 500:
    xsec_test_arr = np.logspace( -23, -15, 25 )
elif mass > 500 and mass <= 1500:
    xsec_test_arr = np.logspace( -22, -14, 25  )
elif mass > 1500:
    xsec_test_arr = np.logspace( -20, -12, 25  )
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
            print(ib, len(SCD_params_xsec_ebins))
            ix = np.argmin(np.abs(SCD_params_xsec_ebins[ib] - xsec_t))

            ## FLOATING NORM, MID SLOPE

            Fb1 = (np.array(Fb_ebins_xsec[ib][ix])*((xsec_t/SCD_params_xsec_ebins[ib][ix])))[0]
            Fb2 = (np.array(Fb_ebins_xsec[ib][ix])*((xsec_t/SCD_params_xsec_ebins[ib][ix])))[1]
            Fb3 = (np.array(Fb_ebins_xsec[ib][ix])*((xsec_t/SCD_params_xsec_ebins[ib][ix])))[2]

            scipy_min = minimize( lambda args: ll_func( xsec_t, ix, args[0], args[1], args[2], -2-args[3], blazar_SCD[ib][1], -3+args[4], blazar_SCD[ib][3], blazar_SCD[ib][4], blazar_SCD[ib][5], A_ebins_xsec[ib][ix]/((xsec_t/SCD_params_xsec_ebins[ib][ix])), Fb1, Fb2, Fb3 ),
                                  [0.1, 0.1, 0.1, -2-blazar_SCD[ib][0], 3+blazar_SCD[ib][2] ], bounds = [ [0,10], [0,10], [0,10], [0,8], [0,6] ], method="L-BFGS-B", options={'maxiter':10000, 'ftol': 1e-10, 'eps':1e-5, 'disp':True} ) 
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
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
