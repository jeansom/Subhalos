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

nside = 128
floatsig = sys.argv[1] == "True"
poiss = sys.argv[2] == "True"
minuit = sys.argv[3] == "True"

ebins = 2*np.logspace(-1,3,41)[0:41]
my_iebins = [int(sys.argv[4]), int(sys.argv[5])]

xsec_inj = float(sys.argv[6])
mass = 100
mass_inj = 100

name = "NFW"+str(floatsig) + "" + str(poiss)+""+ str(minuit) + "" + str(xsec_inj) + "" + str(my_iebins[0]) + "-" + str(my_iebins[1])
chainsname = "NFW"+str(int(floatsig)) + str(int(poiss)) + str(int(minuit)) + str(int(-np.log10(xsec_inj))) #+ str(my_iebins[0]) + str(my_iebins[1])

SCD_params = np.array([-5.71613418, -5.52961428, 10., 1.79143877, 10., 1.80451023, 10., 1.80451023, 10., 1.80451023, 2.99124533, 2.38678777, 2.38678777, 2.38])
#SCD_params[:len(my_iebins)-1] = 10**SCD_params[:len(my_iebins)-1]
SCD_params[(len(my_iebins) - 1) * (1 + (2+1)):] = 10**(SCD_params[(len(my_iebins) - 1) * (1 + (2+1)):])
exposure_ebins= []
blazars_ebins = []

for ib, b in enumerate(my_iebins[:-1]):
    fermi_exposure = np.zeros(hp.nside2npix(nside))
    blazars_ebins.append(np.load("/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/blazarMC/blazar_map"+str(b)+"_"+str(my_iebins[ib+1])+"_0.npy"))
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=1
        fermi_exposure += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/exposure'+str(bin_ind)+'.npy')
    fermi_exposure = fermi_exposure / n
    exposure_ebins.append(fermi_exposure)
    
#subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')
print("USING NFW TEMPLATE")
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/NFWTemplate.npy')
subhalos = subhalos/np.mean(subhalos)

xsec0 = 1e-22
subhalo_MC = []
for ib, b in enumerate(my_iebins[:-1]):
    fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map_NFWMW_"+str(sys.argv[7])+"_"+str(b)+"-"+str(my_iebins[ib+1])+".npy")*exposure_ebins[ib]*xsec_inj/xsec0
    fake_data = np.round(np.random.poisson(fake_data)).astype(np.int32)
    subhalo_MC.append(fake_data)

pscmask = np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)
#mask = cm.make_mask_total(band_mask=False)

data_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib]*0 ))

#bkg_arr = [ [[np.ones(len(blazars_ebins[ib]))*exposure_ebins[0], 'pois']], [[np.ones(len(blazars_ebins[ib]))*exposure_ebins[0], 'pois']] ]
bkg_arr = [[], []]

if (not floatsig) and (not poiss): 
    bkg_arr_np = [ [[np.ones(len(blazars_ebins[ib])), 'blaz']], [[np.ones(len(blazars_ebins[ib])), 'blaz']] ]
else: 
    bkg_arr_np = [[], []]

ll_ebins, A_ebins, Fb_ebins, n_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, 2, chainsname, bkg_arr, bkg_arr_np, subhalos, False, floatsig, floatsig, *SCD_params )

#print(ll_ebins[0]([8.5437e-04, 10, -2.82379, 1.6367, 90.4835, 105.958]))

if minuit:
    if not floatsig and (not poiss): 
        minuit_min = iminuit.Minuit(lambda A, Ab, n1b, n2b, n3b, Fb1b, Fb2b: -ll_ebins[0]([Ab, n1b, n2b, n3b, Fb1b, Fb2b]), 
                                    A=0., limit_A=(0., 10.), error_A=1e-2, fix_A=True,
                                    Ab=-6, limit_Ab=(-10, -3), error_Ab=1e-2, n1b=10, limit_n1b=(2.05,30.), error_n1b=.1, n2b=2, limit_n2b=(-3,3), error_n2b=1e-1, n3b=1.4, limit_n3b=(0.1,1.95), error_n3b=1e-2, Fb1b=11.8129, limit_Fb1b=(0,100), error_Fb1b=1, Fb2b=0.5, limit_Fb2b=(0.,1), error_Fb2b=1e-2,
                                    print_level=1)
    elif poiss:
        minuit_min = iminuit.Minuit(lambda A: -ll_ebins[0]([A]), 
                                    A=1e-6, limit_A=(0., 10.), error_A=1e-2,
                                    print_level=1)        
    else: 
        minuit_min = iminuit.Minuit(lambda A, Ab, n1b, n2b, n3b, Fb1b, Fb2b: -ll_ebins[0]([Ab, n1b, n2b, n3b, Fb1b, Fb2b]), 
                                    A=0., limit_A=(0., 100.), error_A=1e-2, fix_A=True,
                                    Ab=-7, limit_Ab=(-10, -3), error_Ab=1e-2, n1b=10, limit_n1b=(2.05,30.), error_n1b=.1, n2b=0, limit_n2b=(-3,3), error_n2b=1e-1, n3b=1.4, limit_n3b=(0.1,1.95), error_n3b=1e-2, Fb1b=11.8129, limit_Fb1b=(0,100), error_Fb1b=1, Fb2b=0.5, limit_Fb2b=(0.,1), error_Fb2b=1e-2,
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
    
