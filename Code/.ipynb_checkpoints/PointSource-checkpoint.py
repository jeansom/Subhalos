import astropy.units as u
import astropy.coordinates as coord
import numpy as np
import healpy as hp

from ConcentrationFunc import *

import constants
import units

nside = 128

# General cosmological constants for Milky Way
G = 43007.1 # Gravitational constant in [(km/s)**2*(kpc/(1e10*M_s))]
H0 = 0.07 # Hubble in [(km/s)/kpc]

# Parameters for MW NFW profile                                                                      
rho_c = 3*H0**2/(8*np.pi*G)*1e10 *units.M_s/units.kpc**3  # Critical density in [1e10*M_s/kpc**3] 

class PointSource:
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
        self.r200 = ( 3 * self.Mvir / (4 * np.pi * 200 * rho_c) )**(1./3.)

    def r_halocentric(self, d_halo, d):
        """ Distance to halo given distance d and angle psi_deg from us                       
        """
        return np.abs(d_halo - d)
    
    def setConc(self, conc, concFunc=None):
        if concFunc != None: self.conc = concFunc
        if conc == "SP": self.conc = c200_SP
        elif conc == "S": self.conc = c200_S
            
    def calcJ(self, conc="SP", concFunc=None):
        self.setConc(conc, concFunc)
        self.c = self.conc(self.Mvir, self.Rvir, self.r200)
        self.r_s = self.r200/self.c
        self.J = self.J_fac_integral() / (units.GeV**2/units.Centimeter**5)