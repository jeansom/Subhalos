import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np 
import pandas as pd
import healpy as hp
import scipy.integrate as integrate

import constants_noh as constants
import units

nside = 128
# General cosmological constants for Milky Way                                                       
G = 43007.1 # Gravitational constant in [(km/s)**2*(kpc/(1e10*M_s))]                                 
H0 = 0.07 # Hubble in [(km/s)/kpc]                                                                   
r0 = 8 # Position of the sun in [kpc]                                                                
r_vir = 213.5 # r200 for MW in [kpc] (taken from 1606.04898)                                         

# Parameters for MW NFW profile                                                                      
rho_c = 3*H0**2/(8*np.pi*G)*1e10 *units.M_s/units.kpc**3  # Critical density in [1e10*M_s/kpc**3] 
gamma_NFW = 1
Omegapix = hp.nside2pixarea(128)

class PointSource_ExactJ():
    def __init__(self, Mvir, Rvir, theta, phi, rsatfac):
        self.Mvir = Mvir*units.M_s
        self.Rvir = Rvir*units.kpc
        self.theta = theta
        self.phi= phi
        self.rsatfac = rsatfac
        self.coord = (coord.Galactocentric(x=self.Rvir/units.kpc*np.cos(self.phi)*np.sin(self.theta) * u.kpc,
                                  y=self.Rvir/units.kpc*np.sin(self.phi)*np.sin(self.theta) * u.kpc,
                                  z=self.Rvir/units.kpc*np.cos(self.theta) * u.kpc,
                                  galcen_distance=constants.r0*u.kpc)).transform_to(coord.Galactic) # Convert to galactic coordinates
        self.pixels = hp.ang2pix(nside, np.pi/2.-np.radians(np.array(self.coord.b)), np.radians(np.array(self.coord.l))) # Subhalo pixel numbers
 
    def PPnoxsec(self, DMmass, ebins, channel, energy=False):
        dNdLogx_df = pd.read_csv('/tigress/somalwar/Subhaloes/Subhalos/Data/AtProduction_gammas.dat', delim_whitespace=True)

        dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(DMmass)))))[['Log[10,x]',channel]]
        Egamma = np.array(DMmass*(10**dNdLogx_ann_df['Log[10,x]']))
        dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
        dNdE_interp = interp1d(Egamma, dNdEgamma)
        if ebins[0] < DMmass:
            if ebins[1] < DMmass:
                # Whole bin is inside
                if energy: PPnoxsec = 1.0/(8*np.pi*DMmass**2)*quad(lambda x: x*dNdE_interp(x), ebins[0], ebins[1])[0];
                else: PPnoxsec = 1.0/(8*np.pi*DMmass**2)*quad(lambda x: dNdE_interp(x), ebins[0], ebins[1])[0];
            else:
                # Bin only partially contained
                if energy: PPnoxsec = 1.0/(8*np.pi*DMmass**2)*quad(lambda x: x*dNdE_interp(x), ebins[0], DMmass)[0];
                else: PPnoxsec = 1.0/(8*np.pi*DMmass**2)*quad(lambda x: dNdE_interp(x), ebins[0], DMmass)[0];
        else: PPnoxsec = 0
        return PPnoxsec

    def r_halocentric(self, d_halo, theta, d):
        """ Distance to halo given distance d and angle psi_deg from us                       
        """
        return np.sqrt(d_halo**2. + d**2. - 2.*d_halo*d*np.cos(theta))

    def rho0_NFW(self, id, gamma=gamma_NFW):
        rho0 = self.Mvir[id] / (4*np.pi*self.r_s[id]**3*( np.log(1 + self.r200[id]/self.r_s[id]) - self.r200[id]/(self.r200[id]+self.r_s[id]) ))
        return rho0
                                              
    def rho_NFW(self, rho0, r, id, gamma=gamma_NFW):
        if r > self.rsat[id]:
            return rho0/( (r/self.r_s[id])**gamma * (1+r/self.r_s[id])**(3-gamma) )
        else:
            return (4*np.pi*rho0*self.r_s[id]**3*( np.log(1 + self.rsat[id]/self.r_s[id]) - self.rsat[id]/(self.rsat[id]+self.r_s[id]) ))/(4/3 * np.pi * self.rsat[id]**3 )
            

    def J_fac_integral_NFW(self):
        """ NFW line of sight integral of rho(r)**2
        """
        J_int = []
        for id, dval in enumerate(self.coord.distance.kpc):
            rho0 = self.rho0_NFW(id)
            d_halo = self.coord.distance.kpc[id]*units.kpc
            l_halo = self.coord.l.radian[id]
            d = np.linspace(0, dval*units.kpc + self.r200[id], 20000000)
            theta = np.linspace(0, np.pi, 10000)
            J_int.append( integrate.quad(lambda d: self.rho_NFW(rho0, self.r_halocentric(d_halo, 0, d), id)**2, max(0,dval*units.kpc-10*self.r200[id]), dval*units.kpc + 10*self.r200[id])[0] )
        return np.array(J_int)

    def calcJ(self, conc="SP", concFunc=None):
        """ For each pixel, get the line of sight integral of
            rho_NFW**2 to get the J-factor map
        """
        if concFunc != None: self.conc = concFunc
        if conc == "SP": self.conc = self.c200_SP
        elif conc == "S": self.conc = self.c200_S
        self.r200 = ( 3 * self.Mvir / (4 * np.pi * 200 * rho_c) )**(1./3.)
        self.c = self.conc(self.Mvir, self.Rvir, self.r200)
        self.r_s = self.r200/self.c
        self.rsat = self.rsatfac*self.r200
        integrand_NFW = self.J_fac_integral_NFW()
        Jfactor_NFW = integrand_NFW # multiply by the solid angle subtended by each pixel to get J(delta Omega)
        self.J = Jfactor_NFW / (units.GeV**2/units.Centimeter**5)
    
    def c200_SP( self, M200, r, r200 ): # Sanchez-Conde-Prada mass-concentration relation
        c_arr = [37.5153, -1.5093, 1.636 * 10**(-2), 3.66 * 10**(-4), -2.89237 * 10**(-5), 5.32 * 10**(-7)]
        c_arr.reverse()
        c200_val = np.polyval(c_arr, np.log(M200*units.h/units.M_s))
        return c200_val #np.log(np.random.lognormal( c200_val, 0.14 ))

    def c200_S( self, M200, r, r200 ): # Distance dependent c-m relation
        alphaR = 0.286
        C1 = 119.75
        C2 = -85.16
        alpha1 = 0.012
        alpha2 = 0.0026
        return (r/(402 * units.kpc))**(-alphaR) * ( C1*(M200/units.M_s)**(-alpha1) + C2*(M200/units.M_s)**(-alpha2))
