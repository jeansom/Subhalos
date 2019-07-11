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
#SCD_params = np.array([-5.43320258, 2.05, 1.76690669, 1.29602111, 60.79630843, 0.1])
SCD_params = np.array([-5.44404671, 2.05, 1.76208101, 1.29979701, 61.41696489, 0.1])
#blazar_SCD = np.array([-4.98375653, 2.56876073, 1.83669319, 1.49565617, 81.12902519, 0.1])
#blazar_SCD = np.array([-5.07112144,  2.58133598,  1.85159109,  1.50325427, 88.67124125,  0.1])
blazar_SCD = np.array([-3.4008801302614367,  2.05,        1.51678111,  1.46127904, 11.84505654,  0.10002035])
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

xsec0 = 1e-22
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')
subhalos = subhalos/np.mean(subhalos)

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
                            [ 0.1, 0.1, 0.1 ], method="SLSQP", bounds = [ [0,20], [0,20], [0,20] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )

        data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib]))
        #data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib], bkg_min.x[0]*dif_ebins[ib], bkg_min.x[1]*iso_ebins[ib], bkg_min.x[2]*psc_ebins[ib] ))
        #data_ebins.append(np.round(np.random.poisson(subhalo_MC[ib])).astype(np.int32))
        data_ebins[-1][data_ebins[-1] > 1000] = 1000
        data_ebins[-1] = np.load("MPITemp/fake_data0_test.npy")
        #np.save("MPITemp/fake_data"+str(ib)+"_"+tag, data_ebins[-1])
        #data_ebins[-1] = np.load("MPITemp/fake_data0_test.npy")
comm.Barrier()
if comm.rank != 0:
    for ib, b in enumerate(my_iebins[:-1]):
        data_ebins.append(np.load("MPITemp/fake_data"+str(ib)+"_"+tag+".npy"))

bkg_arr = []
for ib in range(len(my_iebins)-1):
    bkg_arr.append([])
#    bkg_arr.append([ [ dif_ebins[ib], 'dif'], [iso_ebins[ib], 'iso'], [psc_ebins[ib], 'psc'] ] )

bkg_arr_np = [[[np.ones(len(blazars_ebins[ib])), 'blaz']], [[np.ones(len(blazars_ebins[ib])), 'blaz']]]
#bkg_arr_np = [ [[]], [[]] ]

ll_ebins, A_ebins, Fb_ebins, n_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, bkg_arr_np, subhalos, False, False, True, *SCD_params )
ll_arr = []
SCDb_arr_ebins = []
def ll_func( xsec_t, A_dif, A_iso, A_psc, Ab, n1b, n2b, n3b, Fb1b, Fb2b ): 
    return -ll_ebins[ib]([Ab, n1b, n2b, n3b, Fb1b, Fb2b, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]) # NON-POISSONIAN BACKGROUNDS

#    return -ll_ebins[ib]([A_dif, A_iso, A_psc, Ab, n1b, n2b, n3b, Fb1b, Fb2b, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]) # ALL BACKGROUNDS

#    return -ll_ebins[ib]([A_dif, A_iso, A_psc, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]) # POISSON BACKGROUNDS

#    return -ll_ebins[ib]([A_ebins[ib]/((xsec_t/xsec0)), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0))) ]) # NO BACKGROUNDS

xsec_test_arr = np.logspace(-30, -20, 101)
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

            '''
            ## ALL BACKGROUNDS
            scipy_min = minimize( lambda args: ll_func( xsec_t, *args ),
                                  [0.1, 0.1, 0.1, -6, 5, 0, 0.1, 60, 0.5], method="SLSQP", bounds = [ [0, 20], [0, 20], [0, 20], [-10, -2], [2.05,10], [-3,3], [-10, 1.95], [1,100], [0.1, 1.] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            '''
            ## NON-POISSON BACKGROUNDS
            scipy_min = minimize( lambda args: ll_func( xsec_t, 0, 0, 0, *args ),
                                  [-6, 5, 0, 0.1, 60, 0.5], method="SLSQP", bounds = [ [-10, -2], [2.05,10], [-3,3], [-10, 1.95], [1,100], [0.1, 1.] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            print(scipy_min.x)
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            '''
            ## POISSON BACKGROUND
            scipy_min = minimize( lambda args: ll_func( xsec_t, *args, 0., 1, 1., 0.1, 1000, 0.5 ),
                                  [0.1, 0.1, 0.1], method="SLSQP", bounds = [ [0, 20], [0, 20], [0, 20] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            '''
            ## NO BACKGROUNDS
            ll += -ll_func( xsec_t, 0., 1, 1., 0.1, 1000000., 0.5 )
            SCDb_arr.append([0., 5, 0, 0.1, 60, 0.5])
            '''
            '''
            ## FIX BLAZARS
            #ll += -ll_func( xsec_t, 0, 0, 0, -4.77303859,  2.4897986 ,  1.80402724,  1.43714242, 65.10919031, 0.12527546 )
            #SCDb_arr.append([ -4.77303859,  2.4897986 ,  1.80402724,  1.43714242, 65.10919031, 0.12527546 ])
            ll += -ll_func( xsec_t, 0, 0, 0, *blazar_SCD )
            SCDb_arr.append(blazar_SCD)
            '''

            ## FLOATING NORM
            scipy_min = minimize( lambda args: ll_func( xsec_t, 0, 0, 0, args[0], *blazar_SCD[1:]  ),
                                  [-6], method="SLSQP", bounds = [ [-10, -2] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            print(scipy_min.x)
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))

            '''
            ## FLOATING NORM, MID BREAK
            scipy_min = minimize( lambda args: ll_func( xsec_t, 0, 0, 0, args[0], blazar_SCD[1], args[1], blazar_SCD[3], blazar_SCD[4], blazar_SCD[5] ),
                                  [-6, 0], method="SLSQP", bounds = [ [-10, -2], [-3, 3] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            print(scipy_min.x)
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            print(ll)
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

    
