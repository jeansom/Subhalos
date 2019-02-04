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
parser.add_argument('-u', '--useMC', type=str, help='MC to use')
parser.add_argument('-s', '--useSubhalo', type=str, help='Subhalo MC to use')
parser.add_argument('-r', '--trial', type=float, help='trial number')
parser.add_argument('-m', '--mass', type=float, help='mass to inject')
args = parser.parse_args()

xsec_inj = args.xsec_inj

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
    np.save("fake_data/fake_data"+str(ib)+"_"+tag, data_ebins[-1])
