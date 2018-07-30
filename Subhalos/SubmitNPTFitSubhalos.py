# Import relevant modules
import os
import copy

import pandas as pd
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize
from astropy.io import fits
from tqdm import *
import iminuit
from iminuit import Minuit, describe, Struct

# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis

FermiData = np.load('fermi_data/fermidata_counts.npy').astype(np.int32)
fermi_exposure = np.load('fermi_data/fermidata_exposure.npy')
dif = np.load('fermi_data/template_dif.npy')
iso = np.load('fermi_data/template_iso.npy')
psc = np.load('fermi_data/template_psc.npy')
subhalos = np.load('EinastoTemplate.npy')
subhalos = subhalos

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-x', '--xsec', type=float)
args = parser.parse_args()
xsec_run = args.xsec

fac = xsec_run * 1e22
print(fac)

fake_data = 13.9583217*dif + 1.06289421*iso + 0.90448092*psc
fake_data += np.load("subhalo_flux_map.npy")*fermi_exposure*fac
fake_data = np.random.poisson(fake_data)
fake_data = fake_data.astype(np.int32)

n = nptfit.NPTF(tag='norm')
n.load_data(fake_data, fermi_exposure)

pscmask=np.array(np.load('fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)
n.load_mask(mask)

dif_copy = dif.copy()
iso_copy = iso.copy()
psc_copy = psc.copy()
subhalos_copy = subhalos.copy()

n.add_template(dif_copy, 'dif')
n.add_template(iso_copy, 'iso')
n.add_template(psc_copy, 'psc')

n.add_poiss_model('dif', '$A_\mathrm{dif}$', [0,20], False)
n.add_poiss_model('iso', '$A_\mathrm{iso}$', [0,3], False)
n.add_poiss_model('psc', '$A_\mathrm{psc}$', [0,3], False)

area_mask = len(mask[~mask])/len(mask) * 4*np.pi * (180/np.pi)**2
xsec0 = 1e-22

A0 = 10**(3.63193)/np.average(fermi_exposure[~mask])*area_mask/np.sum(subhalos[~mask])
n20 = 1.93186
n10 = 10
Fb0 = 10**(-7.71429)*np.average(fermi_exposure[~mask])

xsec_arr = np.logspace(-30, -20, 101)
LL_xsec_ary = np.zeros(len(xsec_arr))

for ix, xsec in tqdm_notebook(enumerate(xsec_arr)):
    A = A0 / (xsec/xsec0)
    Fb = Fb0 * (xsec/xsec0)

    new_n = copy.copy(n)
    new_n.add_template(subhalos_copy, 'subhalos', units='PS')
    new_n.add_non_poiss_model('subhalos', 
                           ['$A^\mathrm{ps}_\mathrm{iso}$','$n_1$','$n_2$','$F_b$'],
                              fixed_params=[[0,A], [1,n10], [2,n20], [3, Fb]],
                              units='counts')
    new_n.configure_for_scan()
    minuit_min = iminuit.Minuit(lambda d, i, p: -new_n.ll([d, i, p]) , d=13.9583217, i=1.06289421, p=0.90448092, fix_d=True, fix_i=True, fix_p=True, limit_d=(0.,20.), limit_i=(0.,3.), limit_p=(0.,3.), error_d=1e-1, error_i=1e-1, error_p=1e-1, print_level=0);
    minuit_min.migrad()
    max_LL = -minuit_min.fval
    best_fit_params = [minuit_min.values]
    print("Best Fit Params:", best_fit_params)
    print("Max LL:", max_LL)
    LL_xsec_ary[ix] = max_LL

np.save("LL_xsec_ary-"+str(round(-1*np.log10(xsec_run), 3)), LL_xsec_ary)
