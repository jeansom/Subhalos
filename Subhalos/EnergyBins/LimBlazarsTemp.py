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

emin = 0
emax = 39
nside = 128

# These are the energy bin edges over which the data is defined
ebins = 2*np.logspace(-1,3,41)[emin:emax+2]
my_iebins = [10, 15, 20]

parser = argparse.ArgumentParser(description="Signal Recovery")
parser.add_argument('-n', '--num_breaks', type=int, help='Number of breaks')
parser.add_argument('-d', '--use_data', default=False, action='store_true', help='Use real data')
parser.add_argument('-x', '--xsec_inj', type=float, help='xsec to inject')
parser.add_argument('-p', '--params_SCD', nargs='+', type=float, help='SCD Params')
parser.add_argument('-t', '--tag', type=str, help='tag for NPTFit')
parser.add_argument('-m', '--mass', type=float, help='DM mass to test')
parser.add_argument('-c', '--chi_mass', type=float, help='DM mass to inject')
parser.add_argument('-u', '--useMC', type=str, help='MC to use')
parser.add_argument('-s', '--useSubhalo', type=str, help='Subhalo MC to use')
parser.add_argument('-f', '--flux', default=False, action='store_true', help='A, Fb in terms of flux or counts, default is counts')
parser.add_argument('-pp', '--pp', nargs="+", type=float, help='Initial PPnoxsec')
parser.add_argument('-r', '--trial', type=float, help='trial number')
args = parser.parse_args()

useData = args.use_data #False
xsec_inj = args.xsec_inj #1e-22
Nb = args.num_breaks #1
SCD_params = np.array(args.params_SCD) #10**(3.53399), 10.0, 1.89914, 10**(-7.71429)
SCD_params[:len(my_iebins)-1] = 10**SCD_params[:len(my_iebins)-1]
SCD_params[(len(my_iebins) - 1) * (1 + (Nb+1)):] = 10**(SCD_params[(len(my_iebins) - 1) * (1 + (Nb+1)):])
tag = args.tag #"subs1"
mass = args.mass
mass_inj = args.chi_mass
PPnoxsec0_ebins = args.pp
trial = int(args.trial)
if "d" in args.useMC: useMadeMC = args.useMC
else: useMadeMC = None
if "sub" in args.useSubhalo: useSubhalo = args.useSubhalo
else: useSubhalo = None
isflux = args.flux

exposure_ebins= []
dif_ebins= []
iso_ebins= []
psc_ebins = []
fermi_data_ebins = []
blazars_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    fermi_exposure = np.zeros(hp.nside2npix(nside))
    dif = np.zeros(len(fermi_exposure))
    iso = np.zeros(len(fermi_exposure))
    psc = np.zeros(len(fermi_exposure))
    data = np.zeros(len(fermi_exposure))
    blazars_ebins.append(np.load("blazarMC/blazar_map"+str(b)+"_"+str(my_iebins[ib+1])+"_0.npy"))
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=1
        fermi_exposure += np.load('maps/exposure'+str(bin_ind)+'.npy')
        dif += np.load('maps/dif'+str(bin_ind)+'.npy')
        iso += np.load('maps/iso'+str(bin_ind)+'.npy')
        psc += np.load('maps/psc'+str(bin_ind)+'.npy')
        data += np.load('maps/data'+str(bin_ind)+'.npy')
    fermi_exposure = fermi_exposure / n
    exposure_ebins.append(fermi_exposure)
    dif_ebins.append(dif)
    iso_ebins.append(iso)
    psc_ebins.append(psc)
    fermi_data_ebins.append(data.astype(np.int32))
np.save("datatest", data)

channel = 'b'
dNdLogx_df = pd.read_csv('/tigress/somalwar/Subhaloes/Subhalos/Data/AtProduction_gammas.dat', delim_whitespace=True)
dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(mass)))))[['Log[10,x]',channel]]
Egamma = np.array(mass*(10**dNdLogx_ann_df['Log[10,x]']))
dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
dNdE_interp = interp1d(Egamma, dNdEgamma)
PPnoxsec_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    ebins_temp = [ ebins[b], ebins[my_iebins[ib+1]] ]
    print(ebins_temp)
    if ebins_temp[0] < mass:
        if ebins_temp[1] < mass:
            # Whole bin is inside
            PPnoxsec_ebins.append(1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], ebins_temp[1])[0])
        else:
            # Bin only partially contained
            PPnoxsec_ebins.append(1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], mass)[0])
    else: PPnoxsec_ebins.append(0)
print(PPnoxsec_ebins)
print(PPnoxsec0_ebins)
xsec0 = 1e-22
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')
subhalos = subhalos / np.mean(subhalos)

subhalo_MC = []
if useSubhalo == None: 
    for ib, b in enumerate(my_iebins[:-1]):
        fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map0_"+str(b)+"-"+str(my_iebins[ib+1])+".npy")*exposure_ebins[ib]*xsec_inj/xsec0
        fake_data = np.round(fake_data).astype(np.int32)
        subhalo_MC.append(fake_data)
else: 
    for ib, b in enumerate(my_iebins[:-1]):
        subhalo_MC.append(np.load(useSubhalo+str(b)+"-"+str(my_iebins[ib+1])+".npy")*exposure_ebins[ib]*xsec_inj/xsec0)
np.save("MC_sub", subhalo_MC[0])    
pscmask = np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)

bkgFac = 1.

data_ebins = []
if useData: 
    data = FermiData
elif useMadeMC!=None:
    data_ebins = fermi_data_ebins
else:
    print(my_iebins[:-1])
    for ib, b in enumerate(my_iebins[:-1]):
        print(ib)
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
        minuit_bkg = iminuit.Minuit( lambda d, i, p: -n_bkg.ll([d, i, p]), d=0.1, i=0.1, p=0.1, fix_d=False, fix_i=False, fix_p=False, limit_d=(0.,100.), limit_i=(0.,15.), limit_p=(0.,15.), error_d=1e-1, error_i=1e-1, error_p=1e-1, print_level=1)
        minuit_bkg.migrad()
        best_fit_params = minuit_bkg.values
        data_ebins.append(makeMockData( subhalo_MC[ib], bkgFac*best_fit_params['d']*dif_ebins[ib], bkgFac*best_fit_params['i']*iso_ebins[ib], bkgFac*best_fit_params['p']*psc_ebins[ib], blazars_ebins[ib] ))
        np.save("fake_data"+str(ib), data_ebins[-1])
if useMadeMC==None: np.save("/tigress/somalwar/Subhaloes/Subhalos/MC/mockdata_"+str(tag), data_ebins)
bkg_arr = []
for ib in range(len(my_iebins[:-1])):
    bkg_arr.append([ [ dif_ebins[ib], 'dif'], [iso_ebins[ib], 'iso'], [psc_ebins[ib], 'psc'] ])
bkg_arr_np = []
for ib in range(len(my_iebins[:-1])):
    bkg_arr_np.append([ [ np.ones(len(blazars_ebins[ib])), 'blaz' ] ])

ll_ebins, A_ebins, Fb_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, bkg_arr_np, subhalos, isflux, *SCD_params )
print(A_ebins, Fb_ebins)
xsec_test_arr = np.logspace(-40, -20, 101)
ll_arr = []
i_arr_ebins, d_arr_ebins, p_arr_ebins, SCDb_arr_ebins = [], [], [], []
for xsec_t in xsec_test_arr:
    ll = 0
    for ib in range(len(my_iebins[:-1])):
        i_arr, d_arr, p_arr, SCDb_arr = [], [], [], []
        if PPnoxsec_ebins[ib] != 0:
            def ll_func( d, i, p, Ab, n1b, n2b, n3b, Fb1b, Fb2b ): 
                return -ll_ebins[ib]([d, i, p, Ab, n1b, n2b, n3b, Fb1b, Fb2b, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ])
            minuit_min = iminuit.Minuit(ll_func, 
                                        d=0.9442, i =  0.5767, p =  0.5368, fix_d=False, fix_i=False, fix_p=False, limit_d=(0.,100.), limit_i=(0.,15.), limit_p=(0.,15.), error_d=1e-1, error_i=1e-1, error_p=1e-1, 
#                                        Ab=-6, limit_Ab=(-7,-2), error_Ab=1e-2, n1b=10, limit_n1b=(0.,15.), error_n1b=1., n2b=12, limit_n2b=(-20.,20.), error_n2b=1e-1, n3b=1.4, limit_n3b=(0.,2.), error_n3b=1e-2, Fb1b=12, limit_Fb1b=(5,20), error_Fb1b=1, Fb2b=7.9, limit_Fb2b=(0,10), error_Fb2b=1,
                                        Ab=-6, limit_Ab=(-10,20), error_Ab=1e-2, n1b=3, limit_n1b=(2.05,5), error_n1b=1., n2b=3, limit_n2b=(1.0,3.5), error_n2b=1e-1, n3b=1.4, limit_n3b=(-1.99,1.99), error_n3b=1e-2, Fb1b=12, limit_Fb1b=(0,20), error_Fb1b=1, Fb2b=7.9, limit_Fb2b=(0,20), error_Fb2b=1,
                                        print_level=1)
            minuit_min.migrad()
            ll += (-minuit_min.fval)
            d_arr.append(minuit_min.values['d'])
            i_arr.append(minuit_min.values['i'])
            p_arr.append(minuit_min.values['p'])
            SCDb_arr.append([ minuit_min.values['Ab'], minuit_min.values['n1b'], minuit_min.values['n2b'], minuit_min.values['n3b'], minuit_min.values['Fb1b'], minuit_min.values['Fb2b'] ])
        d_arr_ebins.append(d_arr)
        i_arr_ebins.append(i_arr)
        p_arr_ebins.append(p_arr)    
        SCDb_arr_ebins.append(SCDb_arr)

    ll_arr.append(ll)
ll_arr = np.array(ll_arr)

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
np.savez("lim_fixed_"+str(xsec_inj) + "_" + str(useData) + "_" + str(Nb) + "_" + tag, [xsec_rec, -minuit_min.fval ], ll_arr, d_arr_ebins, i_arr_ebins, p_arr_ebins, SCDb_arr_ebins )
print(xsec_rec, -minuit_min.fval)
