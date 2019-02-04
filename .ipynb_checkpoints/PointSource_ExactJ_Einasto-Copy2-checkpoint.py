import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np 
import pandas as pd
import healpy as hp
import scipy.integrate as integrate

from AssortedFunctions import getPPnoxsec
from ConcentrationFunc import *

import constants
import units

nside = 128

# General cosmological constants for Milky Way
G = 43007.1 # Gravitational constant in [(km/s)**2*(kpc/(1e10*M_s))]
H0 = 0.07 # Hubble in [(km/s)/kpc]
r0 = 8 # Position of the sun in [kpc]
r_vir = 213.5 # r200 for MW in [kpc] (taken from 1606.04898)

# Parameters for MW NFW profile                                                                      
rho_c = 3*H0**2/(8*np.pi*G)*1e10 *units.M_s/units.kpc**3  # Critical density in [1e10*M_s/kpc**3] 
alpha_Ein = 0.17
Omegapix = hp.nside2pixarea(128)

class PointSource_ExactJ_Einasto():
    def __init__(self, Mvir, Rvir, theta, phi):
        self.Mvir = Mvir*units.M_s
        self.Rvir = Rvir*units.kpc
        self.theta = theta
        self.phi= phi
        self.coord = (coord.Galactocentric(x=self.Rvir/units.kpc*np.cos(self.phi)*np.sin(self.theta) * u.kpc,
                                  y=self.Rvir/units.kpc*np.sin(self.phi)*np.sin(self.theta) * u.kpc,
                                  z=self.Rvir/units.kpc*np.cos(self.theta) * u.kpc,
                                  galcen_distance=constants.r0*u.kpc)).transform_to(coord.Galactic) # Convert to galactic coordinates
        self.pixels = hp.ang2pix(nside, np.pi/2.-np.radians(np.array(self.coord.b)), np.radians(np.array(self.coord.l))) # Subhalo pixel numbers

    def r_halocentric(self, d_halo, d):
        """ Distance to halo given distance d and angle psi_deg from us                       
        """
        return np.abs(d_halo - d)

    def rho0_Ein(self, id, alpha=alpha_Ein):
        rho0 = self.Mvir[id] / integrate.quad(lambda r: 4*np.pi*r**2 * np.exp( -2/alpha * ( (r/self.r_s[id])**alpha - 1) ), 0, self.r200[id] )[0]
        return rho0
                                              
    def rho_Ein(self, rho0, r, id, alpha=alpha_Ein):
        return rho0 * np.exp( -2/alpha * ( (r/self.r_s[id])**alpha - 1) )
            
    def J_fac_integral_Ein(self):
        """ NFW line of sight integral of rho(r)**2
        """
        J_int = []
        for id, dval in enumerate(self.coord.distance.kpc):
            rho0 = self.rho0_Ein(id)
            d_halo = self.coord.distance.kpc[id]*units.kpc
            r_arr = np.linspace(1e-50, self.r200[id], 10000000)
            int_arr = 2*np.pi / d_halo**2 * self.rho_Ein(rho0, r_arr, id)**2 / ( 2 * (r_arr/d_halo) ) * np.log( ( (r_arr/d_halo) + 1)**2 / ( (r_arr/d_halo) - 1)**2 )
            J_int.append( integrate.quad(lambda r: 2*np.pi / d_halo**2 * r**2 * self.rho_Ein(rho0, r, id)**2 / ( 2 * (r/d_halo) ) * np.log( ( (r/d_halo) + 1)**2 / ( (r/d_halo) - 1)**2 ), 0, self.r200[id])[0] )
        return np.array(J_int)

    def calcJ(self, conc="SP", concFunc=None):
        if concFunc != None: self.conc = concFunc
        if conc == "SP": self.conc = self.c200_SP
        elif conc == "S": self.conc = self.c200_S

        self.r200 = ( 3 * self.Mvir / (4 * np.pi * 200 * rho_c) )**(1./3.)
        self.c = self.conc(self.Mvir, self.Rvir, self.r200)
        self.r_s = self.r200/self.c
        integrand_Ein = self.J_fac_integral_Ein()
        Jfactor_Ein = integrand_Ein 
        self.J = Jfactor_Ein / (units.GeV**2/units.Centimeter**5)
