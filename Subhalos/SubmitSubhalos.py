import os, sys
import healpy as hp
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import *
from iminuit import Minuit, describe, Struct

# My Functions
import AssortedFunctions
from AssortedFunctions import myLog
import InverseTransform
import PointSource
import PowerLaw

# Siddharth and Laura's Stuff
from NPTFit import create_mask as cm
import constants_noh as constants
import units

channel = 'b'
Nb = 1
conc = "SP"
xsec = 1e-25
marr = [100]
ebins = [2,20]

# Setting basic parameters
nside = 128
npix = hp.nside2npix(nside)
   
pscmask=np.array(np.load('fermi_data/fermidata_pscmask.npy'), dtype=bool)
exposure=np.array(np.load('fermi_data/fermidata_exposure.npy'))
mask = cm.make_mask_total(band_mask = False )
area_rat = (len(mask[~mask]))/len(mask)

# Defining some constants
r_s = 199 # scale radius, [kpc]
alpha = 0.678
N_calib = 150. # Number of subhalos with masses 10^8 - 10^10 M_sun
M_MW = 1.1e12 # [M_s]
mMin_calib = 1e8 # [M_s]
mMax_calib = 1e10 # [M_s]
mMin = 1e-5
mMax = 0.01*M_MW # [M_s]

N_subs = int(1e8)

m_arr = np.logspace(np.log10(mMin), np.log10(mMax), 8000) # mass values to test
def mCDFInv(r):
    return ( m_arr[0]**(-.9) - (m_arr[0]**(-.9) - m_arr[-1]**(-.9))*r)**(-1/.9)

rho_s2 = float(N_subs) / quad(lambda x: 4 * np.pi * x**2 * np.exp( -2./alpha * ( (x/r_s)**(alpha) - 1)), 0, constants.r_vir)[0]
def rho_Ein(r): # Einasto density Profile
    return rho_s2 * np.exp( (-2./alpha) * ( (r/r_s)**(alpha) - 1))

r_arr = np.logspace(0, np.log10(260), 60500) # radius values to test
r_sampler = InverseTransform.InverseTransform(lambda r: r**2 * rho_Ein(r), r_arr, nsamples=N_subs)
m_sampler = InverseTransform.InverseTransform(lambda x: 1, m_arr, nsamples=N_subs, cdfinv=mCDFInv)

PS_set = PointSource.PointSource(m_sampler.sample(), r_sampler.sample(), (np.arccos(2*np.random.rand(N_subs)-1)), (2*np.pi*np.random.rand(N_subs)))
PS_set.calcJ(conc)

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-t', '--trial', type=str)
args = parser.parse_args()
trial = args.trial
np.save("PS_J/PS_J"+str(trial), np.array(PS_set.J.value))
