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
'''
#SCD_params = np.array([-5.44404671, 2.05, 1.76208101, 1.29979701, 61.41696489, 0.1])
SCD_params_arr = [ #np.array([-6.32197606, 2.61152455, 1.90137951, 1.50696229, 58.5387169, 0.13728553 ]),
                   np.array([-5.41424955e+00, 2.17210435e+00, 1.67422471e+00, 1.21940734e+00, 6.11800503e+01, 5.31244704e-02 ]),
#                   np.array([-5.04407790e+00, 2.26433929e+00, 1.37329237e+00, 6.01673619e-01, 1.18546847e+02, 6.91096321e-02 ]),
                   np.array([-4.56940104e+00, 2.06916845e+00, 6.92216318e-01, -1.38899337e+00, 1.25724568e+02, 6.77358398e-02 ])
               ]

#SCD_params_arr = [                    np.array([-5.04407790e+00, 2.26433929e+00, 1.37329237e+00, 6.01673619e-01, 1.18546847e+02, 6.91096321e-02 ]) ]
SCD_params_xsec = np.array([ 1e-22, 1e-20 ])

#SCD_params_xsec = np.array([ 1e-23, 1e-22, 1e-21, 1e-20 ])
'''

SCD_params_arr = [
    np.array([-5.41424955e+00, 2.17210435e+00, 1.67422471e+00, 1.21940734e+00, 6.11800503e+01, 5.31244704e-02 ]),   
    np.array([ -4.98619637477781 , 2.24016924 , 1.36771901 , 0.598503459 , 106.0815055 , 0.06800803089999999 ]),
    np.array([ -5.066038999192546 , 2.3051384099999996 , 1.30267301 , 0.2820041945 , 140.805936 , 0.0434402163 ]),
    np.array([ -4.76502241 , 2.22269868 , 1.14546229 , 0.125770298 , 124.07591 , 0.0570153848 ]),
    np.array([ -4.57571879 , 2.05 , 1.03360119 , -0.374956899 , 86.4358784 , 0.0592700589 ]),
    np.array([ -4.57134424 , 2.05 , 0.957247191 , -0.752547573 , 97.4831177 , 0.057703257 ]),
    np.array([ -4.51326846 , 2.05 , 0.835773901 , -1.03445536 , 97.8904608 , 0.0632892776 ]),
    np.array([ -4.5594701 , 2.05820366 , 0.754631227 , -1.68317832 , 114.956968 , 0.0595772847 ]),
    np.array([ -4.54236738 , 2.07019124 , 0.632113389 , -1.32596527 , 125.187277 , 0.071898825 ]),
    np.array([ -4.55208903 , 2.07045348 , 0.521428991 , -1.39704795 , 137.859196 , 0.0823514823 ]),
    np.array([ -4.59654015 , 2.12875567 , 0.425414999 , -1.3690295 , 159.795016 , 0.0914130897 ]),
    np.array([ -4.61134188 , 2.13100741 , 0.245861525 , -1.38036074 , 172.009221 , 0.103224933 ]),
    np.array([ -4.59718001 , 2.0990278 , 0.0593279958 , -1.33383551 , 176.63565 , 0.117649574 ]),
    np.array([ -4.58431185 , 2.06655174 , -0.549765447 , -0.56790069 , 170.16068346 , 0.77286879 ]),
    np.array([ -4.63732532 , 2.05031271 , -0.857080848 , -0.68655096 , 184.244227 , 0.81489157 ]),
    np.array([ -4.68435333 , 2.06245759 , -1.47686807 , -0.89002684 , 186.42380448 , 0.83344079 ]),
    np.array([ -4.714422515649848 , 2.05 , -1.572507775 , -1.443929775 , 183.09088777 , 0.8553081505 ]),
    np.array([ -4.8692431 , 2.05007773 , -0.74783313 , -1.62840241 , 215.00913456 , 1.0 ]),
    np.array([ -4.990738263317503 , 2.0500000050000002 , -0.61566255 , -2.58494543 , 217.56453428999998 , 1.0 ]),
    np.array([ -5.14560829 , 2.08266773 , 0.229910002 , -3.22390323 , 247.07739611 , 1.0 ]),
    np.array([ -5.30171833 , 2.0566429 , 0.61784026 , -4.28141848 , 253.63927277 , 1.0 ]),
    np.array([ -5.52372047 , 2.17015404 , -0.57010827 , -6.02989334 , 290.31426569 , 1.0 ]),
    np.array([ -5.719179252641322 , 2.0947778699999997 , 0.54204283 , -9.313187500000002 , 279.28840021999997 , 1.0 ]),
    np.array([ -5.9755476396314595 , 2.0929481699999997 , 0.312241114 , -10.0 , 290.06401884 , 1.0 ]),
    np.array([ -6.28815968 , 2.05901097 , -0.237665732 , -10.0 , 289.02456454 , 1.0 ]),
    np.array([ -6.6821251451281505 , 2.05 , -1.5401008900000002 , -10.0 , 275.254589945 , 1.0 ]),
    np.array([ -7.2309346 , 2.05 , -2.69479712 , -10.0 , 300.12239076 , 1.0 ])
]

SCD_params_xsec = np.array([
    1e-22,
    1e-21 ,
    1.4125375446227497e-21 ,
    1.9952623149688665e-21 ,
    2.818382931264449e-21 ,
    3.981071705534986e-21 ,
    5.6234132519034906e-21 ,
    7.943282347242789e-21 ,
    1.1220184543019562e-20 ,
    1.5848931924611108e-20 ,
    2.238721138568347e-20 ,
    3.162277660168379e-20 ,
    4.4668359215096164e-20 ,
    6.309573444801891e-20 ,
    8.912509381337441e-20 ,
    1.2589254117941713e-19 ,
    1.7782794100389227e-19 ,
    2.5118864315095717e-19 ,
    3.548133892335731e-19 ,
    5.011872336272715e-19 ,
    7.079457843841402e-19 ,
    1e-18 ,
    1.4125375446227497e-18 ,
    1.9952623149688666e-18 ,
    2.818382931264449e-18 ,
    3.981071705534985e-18 ,
    5.623413251903491e-18
])

#blazar_SCD = np.array([-4.98375653, 2.56876073, 1.83669319, 1.49565617, 81.12902519, 0.1])
#blazar_SCD = np.array([-5.07112144,  2.58133598,  1.85159109,  1.50325427, 88.67124125,  0.1])
#blazar_SCD = np.array([-3.4008801302614367,  2.05,        1.51678111,  1.46127904, 11.84505654,  0.10002035])
blazar_SCD = np.array([-4.733327518453883, 2.4461263, 1.77293727, 1.48618555, 60.55017686, 0.1 ])
#blazar_SCD = np.array([0, 2.4461263, 1.77293727, 1.48618555, 60.55017686, 0.1 ])
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
        fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map_NFW_"+str(b)+"-"+str(my_iebins[ib+1])+"_"+str(SCD_params_xsec[np.argmin(np.abs(SCD_params_xsec - xsec_inj))])+".npy")*exposure_ebins[ib]*xsec_inj/SCD_params_xsec[np.argmin(np.abs(SCD_params_xsec - xsec_inj))]
#        fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map0_"+str(b)+"-"+str(my_iebins[ib+1])+"_.npy")*exposure_ebins[ib]*xsec_inj/xsec0
        fake_data = np.round(np.random.poisson(fake_data)).astype(np.int32)
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
        data_ebins[-1][data_ebins[-1] > 1000] = 0
        np.save("MPITemp/fake_data"+str(ib)+"_"+tag, data_ebins[-1])
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

ll_ebins_xsec = []
A_ebins_xsec = []
Fb_ebins_xsec = []
n_ebins_xsec = []
for ix in range(len(SCD_params_arr)):
    ll_ebins, A_ebins, Fb_ebins, n_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, bkg_arr_np, subhalos, False, False, True, *SCD_params_arr[ix] )
    ll_ebins_xsec.append(ll_ebins)
    A_ebins_xsec.append(A_ebins)
    Fb_ebins_xsec.append(Fb_ebins)
    n_ebins_xsec.append(n_ebins)

ll_arr = []
SCDb_arr_ebins = []
def ll_func( xsec_t, ix, A_dif, A_iso, A_psc, Ab, n1b, n2b, n3b, Fb1b, Fb2b ): 
    return -ll_ebins_xsec[ix][ib]([Ab, n1b, n2b, n3b, Fb1b, Fb2b, A_ebins_xsec[ix][ib]/((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]) # NON-POISSONIAN BACKGROUNDS

#    return -ll_ebins[ib]([A_dif, A_iso, A_psc, Ab, n1b, n2b, n3b, Fb1b, Fb2b, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]) # ALL BACKGROUNDS

#    return -ll_ebins[ib]([A_dif, A_iso, A_psc, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]) # POISSON BACKGROUNDS
#    return -ll_ebins_xsec[ix][ib]([A_ebins_xsec[ix][ib]/((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]))]) # NO BACKGROUNDS

xsec_test_arr = np.logspace(-30, -15, 101)
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

            ## FIX BLAZARS
            #ll += -ll_func( xsec_t, 0, 0, 0, -4.77303859,  2.4897986 ,  1.80402724,  1.43714242, 65.10919031, 0.12527546 )
            #SCDb_arr.append([ -4.77303859,  2.4897986 ,  1.80402724,  1.43714242, 65.10919031, 0.12527546 ])

            ix = np.argmin(np.abs(SCD_params_xsec - xsec_t))
            print( xsec_t, SCD_params_xsec[ix] )
            ll += -ll_func( xsec_t, ix, 0, 0, 0, *blazar_SCD )

            #ll_add = -np.inf
            #ix_f = -1
            #for ix in range(len(SCD_params_xsec)):
            #    ll_test = -ll_func( xsec_t, ix, 0, 0, 0, *blazar_SCD )
            #    if ll_test > ll_add: 
            #        ll_add = ll_test
            #        ix_f = ix
            #ll += ll_add
            #print(xsec_t, SCD_params_xsec[ix_f])
            #SCDb_arr.append(blazar_SCD)

            '''
            ## FLOATING NORM
            scipy_min = minimize( lambda args: ll_func( xsec_t, 0, 0, 0, args[0], *blazar_SCD[1:]  ),
                                  [-6], method="SLSQP", bounds = [ [-10, -2] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            print(scipy_min.x)
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
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

    
