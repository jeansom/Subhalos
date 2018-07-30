import os
import copy

import pandas as pd
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize

# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis

fermi_exposure = np.load('fermi_data/fermidata_exposure.npy')
subhalos = np.load('MC/EinastoTemplate.npy')
subhalos = subhalos

fake_data = np.load("MC/subhalo_flux_map0.npy")*fermi_exposure
fake_data = np.round(fake_data).astype(np.int32)
#np.save("fake_data", fake_data)

LL_xsec_ary_arr = []
new_n_arr = []
d_arr_ary = []
n = nptfit.NPTF(tag='test22_fn_100')
n.load_data(fake_data, fermi_exposure)
pscmask=np.array(np.load('fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)
n.load_mask(mask)

subhalos_copy = subhalos.copy()

area_mask = len(mask[~mask])/len(mask) * 4*np.pi * (180/np.pi)**2
xsec0 = 1e-22

A0 = 10**(3.53399)/np.average(fermi_exposure[~mask])*area_mask/np.sum(subhalos[~mask])
n20 = 1.89914
n10 = 10.0
Fb0 = 10**(-7.71429)*np.average(fermi_exposure[~mask])

n.add_template(subhalos_copy, 'subhalos', units='PS')
n.add_non_poiss_model('subhalos', 
                       ['$A^\mathrm{ps}_\mathrm{iso}$','$n_1$','$n_2$', '$F_b1$'],
                       [[-10, 1], [-3., 3.]],
                        [True, False],
                          fixed_params=[[1, n10], [3, Fb0]],
                          units='counts')

n.configure_for_scan()
n.perform_scan(nlive=100)
