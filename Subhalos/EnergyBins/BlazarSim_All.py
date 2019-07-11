
# coding: utf-8

# In[14]:

import os, sys
import copy
import healpy as hp
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from tqdm import *
import iminuit
from iminuit import Minuit, describe, Struct
from scipy.interpolate import interp1d
from scipy.optimize import minimize

sys.path.append("/tigress/somalwar/Subhaloes/Subhalos/Modules/")
sys.path.append("/tigress/somalwar/Subhaloes/Subhalos/")
# My Functions"
import AssortedFunctions
from AssortedFunctions import myLog
import InverseTransform
import PowerLaw

# Siddharth and Laura's Stuff
import constants_noh as constants
import units

# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis

# Global settings
nside=128
emin = 0
emax = 39

channel = 'b'
ebins = 2*np.logspace(-1,3,41)[emin:emax+2]
my_iebins = [10, 15]
print(ebins[my_iebins])


exposure_ebins= []
for ib, b in enumerate(my_iebins[:-1]):
    fermi_exposure = np.zeros(len(np.load("maps/exposure0.npy")))
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=1
        fermi_exposure += np.load("maps/exposure"+str(bin_ind)+".npy")
    fermi_exposure = fermi_exposure / n
    exposure_ebins.append(fermi_exposure)

# Setting basic parameters
nside = 128
npix = hp.nside2npix(nside)
   
pscmask=np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)
area_rat = (len(mask[~mask]))/len(mask)

best_fit_params = []
blaztemp = np.ones(hp.nside2npix(128))
for ib in range(len(my_iebins)-1):
    best_fit_params.append([])
    flux_map_ave = np.zeros(hp.nside2npix(nside))
    flux_map = np.zeros(hp.nside2npix(nside))
    for i in range(100):
        flux_map += np.load("/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/blazarMC/blazar_map_test_10_15_"+str(i)+".npy")
    n = nptfit.NPTF(tag='fit')
    sig = np.round(np.random.poisson(flux_map * exposure_ebins[ib])).astype(np.int32)
    sig = (sig).astype(np.int32)
    n.load_data(sig.copy(), exposure_ebins[ib].copy())
    n.load_mask(mask)
    
    subhalos_copy = blaztemp.copy()
    n.add_template(subhalos_copy, 'subhalos', units='PS')
    n.add_non_poiss_model( 'subhalos',
                           ['$A^\mathrm{ps}_\mathrm{iso}$','$n_1$','$n_2$', '$n_3$', '$S_b1$', '$S_b2$'],
                           [[-10, -1],[2.05, 10],[-3, 3],[-10, 1.95],[0.1,100], [0.1,1.]],
                           [True,False,False, False, False, False],
                           dnds_model='specify_relative_breaks' )
    
    n.configure_for_scan();
    def ll(args):
        A, n1, n2, n3, Fb1, Fb2 = args
        return -n.ll([A, n1, n2, n3, Fb1, Fb2])
    init = [-3, 10, 0, -3, 50, 0.1 ]
    #init = [-4.60156174, 2.06682464, 2.41183233, -0.79960145, 99.98482622, 0.43373899]
    scipy_min = minimize( ll, init, method="SLSQP", bounds = [ [-10, -2], [2.05,10], [-3, 3], [-10, 1.95], [1,100], [0.1, 1.] ], options={'ftol':1e-20, 'eps':1e-5, 'maxiter':5000000, 'disp':True} )
    print(scipy_min.x)
