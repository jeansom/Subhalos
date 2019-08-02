#!/usr/bin/env python
cur_dir = '/tigress/somalwar/SubhalosFresh/100GeV_Plan/'

# Tigress dirs
import os, sys
import copy

import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
from scipy.integrate import quad
from tqdm import *
import json

import matplotlib.pyplot as plt

# NPTFit modules
from NPTFit_nogsl import create_mask as cm

sys.path.append(cur_dir)
sys.path.append(cur_dir+"/Code/")
# My Modules
from Recxsec_modules_NP_psf import getNPTFitLL
from AssortedFunctions import getEnergyBinnedMaps, getPPnoxsec
# NPTFit modules
sys.path.append("/tigress/somalwar/NPTFit_nogsl")
from NPTFit_nogsl import nptfit # module for performing scan
from NPTFit_nogsl import create_mask as cm # module for creating the mask
from NPTFit_nogsl import psf_correction as pc # module for determining the PSF correction
from NPTFit_nogsl import dnds_analysis
from NPTFit_nogsl import psf_correction as pc

nside = 128
npix = hp.nside2npix(nside)
ebins = 2*np.logspace(-1,3,41)[0:41]
my_iebins = [10, 20]

tag = 'script_100'

SCDp_a, SCDx_a, SCDblaz_a = [], [], []
for ib in range(len(my_iebins)-1):
    bfp_a = np.load(cur_dir+"/SCD/blazSCD_100_iebins-"+str(my_iebins[ib])+"to"+str(my_iebins[ib+1])+".npy") # Blazar SCD fit params
    SCDblaz_a.append(bfp_a)

exp_a = getEnergyBinnedMaps('/maps/exposure', cur_dir, my_iebins, mean_exp=None, ave=True, int32=False, nside=nside) # Gets exposure maps

pscmask_a = np.array(np.load(cur_dir+'/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask_a = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180)#, custom_mask = pscmask_a)

data_a = [np.load("../chains/chain_blazcorr_m-10.0GeV_x-0.0_0_datapois.npy")] # data array, smoothed blazars

# Define parameters that specify the Fermi-LAT PSF at 2 GeV
fcore = 0.748988248179
score = 0.428653790656
gcore = 7.82363229341
stail = 0.715962650769
gtail = 3.61883748683
spe = 0.00456544262478

# Define the full PSF in terms of two King functions
def king_fn(x, sigma, gamma):
    return 1./(2.*np.pi*sigma**2.)*(1.-1./gamma)*(1.+(x**2./(2.*gamma*sigma**2.)))**(-gamma)

def Fermi_PSF(r):
    return fcore*king_fn(r/spe,score,gcore) + (1-fcore)*king_fn(r/spe,stail,gtail)

pc_inst = pc.PSFCorrection(delay_compute=True)
pc_inst.psf_r_func = lambda r: Fermi_PSF(r)
pc_inst.sample_psf_max = 10.*spe*(score+stail)/2.
pc_inst.psf_samples = 10000
pc_inst.psf_tag = 'Fermi_PSF_2GeV'
pc_inst.make_or_load_psf_corr()

# Extract f_ary and df_rho_div_f_ary as usual
f_ary = pc_inst.f_ary
df_rho_div_f_ary = pc_inst.df_rho_div_f_ary

n = nptfit.NPTF(tag=tag)
n.load_data(data_a[0], np.ones(len(exp_a[0]))*np.mean(exp_a[0])) # Use average exposure
n.load_mask(mask_a)
n.add_template(np.ones(npix), "blaz", units='PS')
n.add_non_poiss_model( "blaz",
                       ['$A$','$n_1$','$n_2$', '$n_3$', '$S_{b1}$', '$S_{b2}$'],
                       [[(np.log10(SCDblaz_a[0][0])-1), (np.log10(SCDblaz_a[0][0])+1)],[3, 20],[-3,3],[-20, 1.95],[0.1,150],[-3,0]],
                       [True, False, False, False, False, True],
                          dnds_model='specify_relative_breaks')
n.configure_for_scan(f_ary, df_rho_div_f_ary, nexp=1)
n.perform_scan(nlive=100)
