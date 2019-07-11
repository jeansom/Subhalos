# Tigress dirs
import os, sys
import copy

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymultinest

import iminuit
from iminuit import Minuit, describe, Struct
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import basinhopping

# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis
import pandas as pd
import healpy as hp

# My Modules
from Recxsec_modules_NP import makeMockData, getNPTFitLL, SCDParams_Flux2Counts

nside = 128

ebins = 2*np.logspace(-1,3,41)[0:41]
my_iebins = [int(sys.argv[1]), int(sys.argv[2])]

mass = 100
mass_inj = 100

name = "BlazarInj"
chainsname = "BlazarInj"

SCD_params = np.array([-5.71613418, -5.52961428, 10., 1.79143877, 10., 1.80451023, 10., 1.80451023, 10., 1.80451023, 2.99124533, 2.38678777, 2.38678777, 2.38])
SCD_params[(len(my_iebins) - 1) * (1 + (2+1)):] = 10**(SCD_params[(len(my_iebins) - 1) * (1 + (2+1)):])

exposure_ebins= []
blazars_ebins = []

for ib, b in enumerate(my_iebins[:-1]):
    fermi_exposure = np.zeros(hp.nside2npix(nside))
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=1
        fermi_exposure += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/exposure'+str(bin_ind)+'.npy')
    blazars_ebins.append(np.load("/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/blazarMC/blazar_map_nocut_"+str(b)+"_"+str(my_iebins[ib+1])+"_"+str(sys.argv[3])+".npy") * fermi_exposure)
    fermi_exposure = fermi_exposure / n
    exposure_ebins.append(fermi_exposure)
    
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')
subhalos = subhalos/np.mean(subhalos)

pscmask = np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)

data_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    data_ebins.append(makeMockData( np.zeros(len(blazars_ebins[ib])), blazars_ebins[ib] ) )
np.save("data", data_ebins[0])
bkg_arr = [[], []]

bkg_arr_np = [ [[np.ones(len(blazars_ebins[ib])), 'blaz']], [[np.ones(len(blazars_ebins[ib])), 'blaz']] ]

ll_ebins, A_ebins, Fb_ebins, n_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, 2, chainsname, bkg_arr, bkg_arr_np, subhalos, False, False, False, *SCD_params )

def ll(args):
    Ab, n1b, n2b, n3b, Fb1b, Fb2b = args
    return -ll_ebins[0]([Ab, n1b, n2b, n3b, Fb1b, Fb2b])

def print_fun(x, f, accepted):
    print(x)
    print("at minimum %.4f, accepted %d" % (f, int(accepted)))

scipy_min = basinhopping( ll, [-10, 5, 0, 0.1, 60, 0.5], niter=10, minimizer_kwargs = {"method":"SLSQP","bounds":[ [-15, -2], [2.05,10], [-3,3], [-10, 0.95], [1,1000], [0.1, 1.] ],"options":{'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True}}, callback=print_fun )
print(scipy_min.x)
print(-scipy_min.fun)    
