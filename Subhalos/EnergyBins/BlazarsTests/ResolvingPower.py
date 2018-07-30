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

floatsig = sys.argv[1] == "True"
minuit = sys.argv[2] == "True"
print(minuit)
ebins = 2*np.logspace(-1,3,41)[0:41]
my_iebins = [int(sys.argv[3]), int(sys.argv[4])]

xsec_inj = float(sys.argv[5])
mass = 100
mass_inj = 100

name = ""+str(floatsig) + "" + str(minuit) + "" + str(xsec_inj) + "" + str(my_iebins[0]) + "-" + str(my_iebins[1])
chainsname = ""+str(int(floatsig)) + str(int(minuit)) + str(int(-np.log10(xsec_inj))) + str(my_iebins[0]) + str(my_iebins[1])

SCD_params = np.array([-5.71613418, -5.52961428, 10., 1.79143877, 10., 1.80451023, 10., 1.80451023, 10., 1.80451023, 2.99124533, 2.38678777, 2.38678777, 2.38])
SCD_params[:len(my_iebins)-1] = 10**SCD_params[:len(my_iebins)-1]
SCD_params[(len(my_iebins) - 1) * (1 + (2+1)):] = 10**(SCD_params[(len(my_iebins) - 1) * (1 + (2+1)):])
exposure_ebins= []
blazars_ebins = []

for ib, b in enumerate(my_iebins[:-1]):
    fermi_exposure = np.zeros(hp.nside2npix(128))
    blazars_ebins.append(np.load("/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/blazarMC/blazar_map"+str(b)+"_"+str(my_iebins[ib+1])+"_0.npy"))
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=1
        fermi_exposure += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/exposure'+str(bin_ind)+'.npy')
    fermi_exposure = fermi_exposure / n
    exposure_ebins.append(fermi_exposure)

subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')
subhalos = subhalos/np.mean(subhalos)

xsec0 = 1e-22
subhalo_MC = []
for ib, b in enumerate(my_iebins[:-1]):
    fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map0_"+str(b)+"-"+str(my_iebins[ib+1])+".npy")*exposure_ebins[ib]*xsec_inj/xsec0
    fake_data = np.round(fake_data).astype(np.int32)
    subhalo_MC.append(fake_data)

pscmask = np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)

data_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib]*0 ))

bkg_arr = [[], []]
if not floatsig: 
    bkg_arr_np = [ [[np.ones(len(blazars_ebins[ib])), 'blaz']], [[np.ones(len(blazars_ebins[ib])), 'blaz']] ]
else: 
    bkg_arr_np = [[], []]

ll_ebins, A_ebins, Fb_ebins, n_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, 2, chainsname, bkg_arr, bkg_arr_np, subhalos, False, True, floatsig, *SCD_params )
if minuit:
    if not floatsig: 
        minuit_min = iminuit.Minuit(lambda Ab, n1b, n2b, n3b, Fb1b, Fb2b: -ll_ebins[0]([Ab, n1b, n2b, n3b, Fb1b, Fb2b]), 
                                    Ab=1e-6, limit_Ab=(1e-20, 1e-2), error_Ab=1e-2, n1b=10, limit_n1b=(2.05,30.), error_n1b=.1, n2b=2, limit_n2b=(-20,20), error_n2b=1e-1, n3b=1.4, limit_n3b=(0.1,1.95), error_n3b=1e-2, Fb1b=11.8129, limit_Fb1b=(0,30), error_Fb1b=1, Fb2b=8.715, limit_Fb2b=(0.,10), error_Fb2b=1e-2,
                                    print_level=1)
    else: 
        minuit_min = iminuit.Minuit(lambda Ab, n1b, n2b, n3b, Fb1b, Fb2b: -ll_ebins[0]([Ab, n1b, n2b, n3b, Fb1b, Fb2b]), 
                                    Ab=1e-6, limit_Ab=(1e-20, 1e-2), error_Ab=1e-2, n1b=10, limit_n1b=(2.05,30.), error_n1b=.1, n2b=2, limit_n2b=(-20,20), error_n2b=1e-1, n3b=1.4, limit_n3b=(0.1,1.95), error_n3b=1e-2, Fb1b=11.8129, limit_Fb1b=(0,30), error_Fb1b=1, Fb2b=8.715, limit_Fb2b=(0.,10), error_Fb2b=1e-2,
                                    print_level=1)
    minuit_min.migrad()
    print(minuit_min.values)
    print(-minuit_min.fval)
    ll_tot = -minuit_min.fval
    print(ll_tot)
else:
    n_ebins[0].perform_scan(nlive=100)
    n_ebins[0].load_scan()
    an = dnds_analysis.Analysis(n_ebins[0])
    an.make_triangle()
    plt.savefig("Likelihoods/corner_" + name + ".png")
    print(n_ebins[0].a.get_best_fit())
    ll_tot = n_ebins[0].a.get_best_fit()["log_likelihood"]
np.save("Likelihoods/ll_"+name, [ll_tot])
    
