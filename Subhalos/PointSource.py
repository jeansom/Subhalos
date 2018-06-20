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
        self.r200 = ( 3 * self.Mvir / (4 * np.pi * 200 * 1.8788e-26*units.h**2*units.Kilogram/units.Meter**3) )**(1./3.) # Units [kpc]
        self.c = self.conc(self.Mvir, self.Rvir, self.r200)
        
        return self.Mvir**2 * self.c**3/(12*np.pi*self.r200**3)*(1-1/(1+self.c)**3.)*(np.log(1+self.c)-self.c/(1+self.c))**(-2)
        
    def PPnoxsec(self, DMmass, ebins, channel):
        dNdLogx_df = pd.read_csv('Data/AtProduction_gammas.dat', delim_whitespace=True)

        dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(DMmass)))))[['Log[10,x]',channel]]
        Egamma = np.array(DMmass*(10**dNdLogx_ann_df['Log[10,x]']))
        dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
        dNdE_interp = interp1d(Egamma, dNdEgamma)
        if ebins[0] < DMmass:
            if ebins[1] < DMmass:
                # Whole bin is inside
                PPnoxsec = 1.0/(8*np.pi*DMmass**2)*quad(lambda x: dNdE_interp(x), ebins[0], ebins[1])[0];
            else:
                # Bin only partially contained
                PPnoxsec = 1.0/(8*np.pi*DMmass**2)*quad(lambda x: dNdE_interp(x), ebins[0], DMmass)[0];
        return PPnoxsec
        
    def c200_SP( self, M200, r, r200 ): # Sanchez-Conde-Prada mass-concentration relation
        c_arr = [37.5153, -1.5093, 1.636 * 10**(-2), 3.66 * 10**(-4), -2.89237 * 10**(-5), 5.32 * 10**(-7)]
        c200_val = 0
        for i in range(6):
            c200_val += c_arr[i] * ( np.log( M200*units.h/units.M_s ) )**i
        return c200_val

    def c200_S( self, M200, r, r200 ): # Distance dependent c-m relation
        alphaR = 0.286
        C1 = 119.75
        C2 = -85.16
        alpha1 = 0.012
        alpha2 = 0.0026
        return (r/(402 * units.kpc))**(-alphaR) * ( C1*(M200/units.M_s)**(-alpha1) + C2*(M200/units.M_s)**(-alpha2))