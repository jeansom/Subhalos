import numpy as np
import scipy as sp
import healpy as hp
import matplotlib.pyplot as plt

import global_var as gv
import masks
import hists

import logging
logger = logging.getLogger(__name__)

conv_factor = gv.sigma_v * gv.rho_0**2. / (8. * np.pi * gv.m_chi**2.) * gv.photons_per_ann * gv.avg_exposure * 3.08E-5
rho_0_factor = np.power(gv.r_0/gv.r_s, gv.gamma_nfw) * np.power(1. + gv.r_0/gv.r_s, 3. - gv.gamma_nfw)

def rho_dimless(r):
    #r in kpc
    return np.power(r/gv.r_s, -gv.gamma_nfw) * np.power(1. + r/gv.r_s, gv.gamma_nfw - 3.) * rho_0_factor     

def r_NFW(l, psi_deg):
    return np.sqrt(gv.r_0**2. + l**2. - 2.*gv.r_0*l*np.cos(np.radians(psi_deg)))
    
def L_NFW_integral(psi_deg):
    return integrate.quad(lambda l: rho_dimless(r_NFW(l, psi_deg))**2., 0., 100.*gv.r_s)[0]
    
def flux_NFW(psi_deg):
    return conv_factor * 4.*np.pi * L_NFW_integral(psi_deg)/ gv.NPIX
    
def make_NFW_intensity_map():
    psi_deg = np.arange(0., 180.5, 0.5)
    intensity_NFW = conv_factor * np.vectorize(L_NFW_integral)(psi_deg)
    flux_NFW = interpolate.interp1d(psi_deg, intensity_NFW * 4.*np.pi / gv.NPIX)
    psi_deg_pixels = np.array([np.degrees(np.arccos(np.dot([1.0, 0.0, 0.0], hp.pix2vec(gv.NSIDE, pix)))) for pix in range(gv.NPIX)])
    return flux_NFW(psi_deg_pixels)

def simulate_NFW():
    #simulate NFW component
    if gv.do_NFW_sim:
        gv.NFW_um = make_NFW_intensity_map()
        gv.sim_map_NFW = hp.ma([float(np.random.poisson(flux)) for flux in gv.NFW_um])
        gv.sim_map_NFW.mask = np.copy(gv.mask_for_all_maps)
        np.savetxt(gv.sim_dir + 'NFW.dat', gv.sim_map_NFW, '%1.5f')
        logging.info('%s', 'Simulated NFW component and saved to NFW.dat')
    #load previous simulation of NFW component from file
    else:
        gv.sim_map_NFW = hp.ma(np.loadtxt(gv.sim_dir + 'NFW.dat'))
        logging.info('%s', 'Loaded simulated NFW component from ' + gv.sim_dir + 'NFW.dat')             
        gv.sim_map_NFW.mask = np.copy(gv.mask_for_all_maps)
