import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np 
import pandas as pd
import healpy as hp

import constants_noh as constants
import units

nside = 128
# General cosmological constants for Milky Way                                                       
G = 43007.1 # Gravitational constant in [(km/s)**2*(kpc/(1e10*M_s))]                                 
H0 = 0.07 # Hubble in [(km/s)/kpc]                                                                   
r0 = 8 # Position of the sun in [kpc]                                                                
r_vir = 213.5 # r200 for MW in [kpc] (taken from 1606.04898)                                         

# Parameters for MW NFW profile                                                                      
rho_c = 3*H0**2/(8*np.pi*G)*1e10*units.M_s/units.kpc**3  # Critical density in [1e10*M_s/kpc**3] 
print(rho_c)
print(units.M_s)
print(units.h)
class PointSource():
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

    
    def calcJ(self, conc="SP", concFunc=None):
        if concFunc != None: self.conc = concFunc
        if conc == "SP": self.conc = self.c200_SP
        elif conc == "S": self.conc = self.c200_S
        self.J = self.PSJ_int()/(self.coord.distance*units.kpc)**2 / (units.GeV**2/units.Centimeter**5)

    def PSJ_int( self ):
        #rho_c = 1.8788e-26*units.h**2*units.Kilogram/units.Meter**3
        self.r200 = ( 3 * self.Mvir / (4 * np.pi * 200 * rho_c) )**(1./3.)
        self.c = self.conc(self.Mvir, self.Rvir, self.r200)
        self.r_s = self.r200/self.c
        nfw_func = np.log(1+self.c)-self.c/(1+self.c)
        delta_c = 200./3*self.c**3/nfw_func
        rho_s = delta_c*rho_c
        l_s = 4*np.pi/3.*rho_s**2*self.r_s**3
        
        return (1.-(1.+self.r200/self.r_s)**-3)*l_s
        
        #rho0 = self.Mvir / quad( lambda r: r**2 * 4 * np.pi * np.exp( (-2./0.17) * ( (r/r_s)**(0.17) - 1)), 0, self.r200)[0]
        #return quad( lamba r: 4 * np.pi * rho0**2 * (np.exp( (-2./0.17) * ( (r/r_s)**(0.17) - 1) ) )**2, 0, self.r200 )[0]
        ##return self.Mvir**2 * self.c**3/(12*np.pi*self.r200**3)*(1-1/(1+self.c)**3.)*(np.log(1+self.c)-self.c/(1+self.c))**(-2)
        #return self.Mvir**2 * self.c**3/(12*np.pi*self.r200**3)*(7./8.)*(np.log(1+self.c)-self.c/(1+self.c))**(-2)
        
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
        return (r/(260 * units.kpc))**(-alphaR) * ( C1*(M200/units.M_s)**(-alpha1) + C2*(M200/units.M_s)**(-alpha2))