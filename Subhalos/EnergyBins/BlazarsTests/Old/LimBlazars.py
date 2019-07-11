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
from Recxsec_modules import makeMockData, getNPTFitLL, SCDParams_Flux2Counts

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
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=1
        fermi_exposure += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/exposure'+str(bin_ind)+'.npy')
        dif += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/dif'+str(bin_ind)+'.npy')
        iso += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/iso'+str(bin_ind)+'.npy')
        psc += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/psc'+str(bin_ind)+'.npy')
        data += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/data'+str(bin_ind)+'.npy')
    fermi_exposure = fermi_exposure / n
    exposure_ebins.append(fermi_exposure)
    dif_ebins.append(dif)
    iso_ebins.append(iso)
    psc_ebins.append(psc)
    blazars_ebins.append(np.load("/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/blazarMC/blazar_map"+str(b)+"_"+str(my_iebins[ib+1])+"_0.npy") * np.array(fermi_exposure) )
    fermi_data_ebins.append(data.astype(np.int32))

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
        minuit_bkg = iminuit.Minuit( lambda d, i, p: -n_bkg.ll([d, i, p]), d=0.1, i=0.1, p=0.1, fix_d=False, fix_i=False, fix_p=False, limit_d=(0.,100.), limit_i=(0.,15.), limit_p=(0.,15.), error_d=1e-1, error_i=1e-1, error_p=1e-1, print_level=1)
        minuit_bkg.migrad()
        best_fit_params = minuit_bkg.values
        data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib] ))
        np.save("fake_data"+str(ib), data_ebins[-1])
if useMadeMC==None: np.save("/tigress/somalwar/Subhaloes/Subhalos/MC/mockdata_"+str(tag), data_ebins)
bkg_arr = []
for ib in range(len(my_iebins[:-1])):
    bkg_arr.append([ [ dif_ebins[ib], 'dif'], [iso_ebins[ib], 'iso'], [psc_ebins[ib], 'psc'] ] )

ll_ebins, A_ebins, Fb_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, subhalos, isflux, *SCD_params )
print(A_ebins, Fb_ebins)
xsec_test_arr = np.logspace(-40, -20, 101)
ll_arr = []
for xsec_t in xsec_test_arr:
    ll = 0
    for ib in range(len(my_iebins[:-1])):
        i_arr, d_arr, p_arr = [], [], []
        if PPnoxsec_ebins[ib] != 0:
            minuit_min = iminuit.Minuit(lambda d, i, p: -ll_ebins[ib]([d, i, p, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]), d=0., i =  0., p =  0., fix_d=True, fix_i=True, fix_p=True, print_level=1)
            minuit_min.migrad()
            ll += (-minuit_min.fval)
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
np.savez("lim_fixed_"+str(xsec_inj) + "_" + str(useData) + "_" + str(Nb) + "_" + tag, [xsec_rec, -minuit_min.fval ], ll_arr )
print(xsec_rec, -minuit_min.fval)
