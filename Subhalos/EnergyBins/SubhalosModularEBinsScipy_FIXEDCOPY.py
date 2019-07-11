import os, sys
import healpy as hp
import numpy as np
from scipy.integrate import quad
from tqdm import *
import iminuit
from iminuit import Minuit, describe, Struct
from scipy.interpolate import interp1d
from scipy.optimize import minimize

sys.path.append("/tigress/somalwar/Subhaloes/Subhalos/Modules/")
sys.path.append("/tigress/somalwar/Subhaloes/Subhalos/")
# My Functions
import AssortedFunctions
from AssortedFunctions import myLog
import InverseTransform
import PointSource
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

trials = 1
channel = 'b'
Nb = 2
conc = "SP"
xsec = float(sys.argv[2])
marr = [10000]
ebins = 2*np.logspace(-1,3,41)[emin:emax+2]
my_iebins = [10, 15]

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
npix = hp.nside2npix(nside)
   
pscmask=np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)
area_rat = (len(mask[~mask]))/len(mask)

# Defining some constants
r_s = 199 # scale radius, [kpc]
r_s_NFW = 17 # scale radius [kpc]
alpha = 0.678
N_calib = 300. # Number of subhalos with masses 10^8 - 10^10 M_sun
M_MW = 1.1e12 # [M_s]
mMin_calib = 1e8 # [M_s]
mMax_calib = 1e10 # [M_s]
mMin = 1e-5*M_MW
mMax = .01*M_MW # [M_s]
min_flux_arr = []
for exposure in exposure_ebins:
    min_flux_arr.append(np.log10(1./(np.sum(exposure[~mask])/len(exposure[~mask]))))

def dNdm_func(m): # Subhalo mass function
    norm = N_calib / ( -.9**(-1) * (mMax_calib**(-.9) - mMin_calib**(-.9)))
    return norm * (m)**(-1.9)

N_subs = np.random.poisson( round(N_calib / ( -.9**(-1) * (mMax_calib**(-.9) - mMin_calib**(-.9))) * -.9**(-1) * (mMax**(-.9) - mMin**(-.9))) ) # Total number of subhalos

m_arr = np.logspace(np.log10(mMin), np.log10(mMax), 8000) # mass values to test
def mCDFInv(r):
    return ( m_arr[0]**(-.9) - (m_arr[0]**(-.9) - m_arr[-1]**(-.9))*r)**(-1/.9)

rho_s2 = float(N_subs) / quad(lambda x: 4 * np.pi * x**2 * np.exp( -2./alpha * ( (x/r_s)**(alpha) - 1)), 0, 2000)[0]
def rho_Ein(r): # Einasto density Profile
    return rho_s2 * np.exp( (-2./alpha) * ( (r/r_s)**(alpha) - 1))

rho_0_NFW = float(N_subs) / quad(lambda x: 4 * np.pi * x**2 * 1 / ( (x/r_s_NFW) * (1 + x/r_s_NFW)**2 ), 0, 2000)[0]
def rho_NFW(r): # NFW density profile
    return rho_0_NFW / ( (r/r_s_NFW) * (1 + r/r_s_NFW)**2 )

r_arr = np.logspace(0, np.log10(2000), 60500) # radius values to test
r_sampler = InverseTransform.InverseTransform(lambda r: r**2 * rho_Ein(r), r_arr, nsamples=N_subs)
m_sampler = InverseTransform.InverseTransform(dNdm_func, m_arr, nsamples=N_subs, cdfinv=mCDFInv)

PS_arr_ebins = []
F_arr_ebins = []
PPnoxsec_ebins = []
flux_bins = np.logspace(-15, -6, 8*8)
for ib in range(len(my_iebins)-1):
    PS_arr = []
    F_arr = []
    for i in tqdm_notebook(range(trials)):
        rval_arr = r_sampler.sample()
        mval_arr = m_sampler.sample()

        theta_arr = (np.arccos(2*np.random.rand(N_subs)-1))
        phi_arr = (2*np.pi*np.random.rand(N_subs))

        PS_set = PointSource.PointSource(mval_arr, rval_arr, theta_arr, phi_arr)
        PS_set.calcJ(conc)
        PS_arr.append(PS_set)

        if i == 0: 
            PPnoxsec = PS_set.PPnoxsec(marr[0], [ ebins[my_iebins[ib]], ebins[my_iebins[ib+1]] ], channel)
            PPnoxsec_ebins.append(PPnoxsec)

#        PS_set.J.value[ PS_set.J.value * xsec * PPnoxsec * np.mean(exposure_ebins[ib][PS_set.pixels]) > 1000 ] = 0

        F_arr.append(np.histogram(PS_set.J.value[~mask[PS_set.pixels]] * xsec * PPnoxsec, bins=flux_bins)[0])
    PS_arr_ebins.append(PS_arr)
    F_arr_ebins.append(F_arr)

min_flux_ind_arr = []
F_ave_arr = []
max_flux_arr = []
dF_arr = []
dN_arr = []
F_arr_arr = []
F_val_arr = []
for ib in range(len(my_iebins)-1):
    min_flux_ind_arr.append(np.argmin(np.abs(flux_bins - 10.**(min_flux_arr[ib]))))
    flux_bins2 = flux_bins[min_flux_ind_arr[-1]:]
    F_arr = np.array(F_arr_ebins[ib]).astype(float)
    for i in range(len(F_arr)):
        F_arr[i][F_arr[i]<=1e-30] = 1e-50
    F_arr_arr.append(F_arr)
    F_ave_arr.append((np.median(F_arr, axis=0))[min_flux_ind_arr[-1]:])
    max_flux_arr.append(np.log10(flux_bins2[np.argmax(F_ave_arr[-1] < 1e-20)]))
    dF_arr.append(np.diff(flux_bins2))
    dN_arr.append(np.array(F_ave_arr[-1])/(4*np.pi*(180/np.pi)**2*area_rat))
    F_val_arr.append((np.array(flux_bins2)[:-1]+np.array(flux_bins2)[1:])/2.)

print(repr(F_ave_arr))
print(repr(F_val_arr))
print(repr(dN_arr))
print(repr(dF_arr))
print(repr(F_arr_arr))

best_fit_params = []
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')
subhalos = subhalos/np.mean(subhalos)
for ib, PS_arr in enumerate(PS_arr_ebins):
    flux_map_ave = np.zeros(hp.nside2npix(nside))
    for iP, PS_set in (enumerate(PS_arr)):
        flux_map = np.zeros(hp.nside2npix(nside))
        flux = PS_set.J.value * xsec * PPnoxsec_ebins[ib]
        for ipix, pix in enumerate(PS_set.pixels):
            flux_map[pix] += flux[ipix]
        if xsec == 1e-22: np.save("/tigress/somalwar/Subhaloes/Subhalos/MC/FixedSCD/subhalo_flux_map_NFW_10TeV_"+str(sys.argv[1])+"_"+str(my_iebins[ib])+"-"+str(my_iebins[ib+1])+"_"+str(xsec), flux_map) 
        n = nptfit.NPTF(tag='fit')
        sig = np.round(flux_map * np.mean(exposure_ebins[ib])).astype(np.int32)

        n.load_data(sig.copy(), exposure_ebins[ib].copy())
        n.load_mask(mask) 

        subhalos_copy = subhalos.copy()
        n.add_template(subhalos_copy, 'subhalos', units='PS')
        n.add_non_poiss_model( 'subhalos',
                               ['$A^\mathrm{ps}_\mathrm{iso}$','$n_1$','$n_2$', '$n_3$', '$S_b1$', '$S_b2$'],
                               [[-10, -1],[2.05, 10],[-3, 3],[-10.0, 1.95],[1, 5000], [0, 1]],
                               [True,False,False, False, False, False],
                               dnds_model='specify_relative_breaks' )

        n.configure_for_scan();

        def ll(args):
            A, n1, n2, n3, Fb2 = args
            Fb1 = 1000
            print( A, n1, n2, n3, Fb1, Fb2 )
            return -n.ll([A, n1, n2, n3, Fb1, Fb2])
        scipy_min = minimize( ll, [-6, 10, 0, 0.1, 0.5], method="SLSQP", bounds = [ [-10, -2], [2.05, 15], [-3,3], [-10, 1.95], [0.01, 0.1] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
        #scipy_min = minimize( ll, [-6, 5, 0, 0.1, 60, 0.5], method="SLSQP", bounds = [ [-10, -2], [2.05,10], [-3,3], [-10, 1.95], [1,1000], [0.01, 1] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
        #scipy_min = minimize( ll, [-6, 10, 0, 0.1, 1000, 10**(-3)], method="SLSQP", bounds = [ [-10, -2], [2.05, 15], [-3,3], [-10, 1.95], [800,1200], [0.5*1e-3, 1.5*1e-3] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
        max_LL = -scipy_min.fun
        best_fit_params.append( np.array( [scipy_min.x[0], scipy_min.x[1], scipy_min.x[2], scipy_min.x[3], scipy_min.x[4] ]))

        print("Best Fit Params:", best_fit_params)
        print("Max LL:", max_LL)
