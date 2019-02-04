import numpy as np 
from PointSource import PointSource
import units

G = 43007.1 # Gravitational constant in [(km/s)**2*(kpc/(1e10*M_s))]
H0 = 0.07 # Hubble in [(km/s)/kpc]
rho_c = 3*H0**2/(8*np.pi*G)*1e10*units.M_s/units.kpc**3  # Critical density in [1e10*M_s/kpc**3] 

class PS(PointSource):
    def __init__(self, Mvir, Rvir, theta, phi):
        PointSource.__init__(self, Mvir, Rvir, theta, phi)

    def J_fac_integral( self ):
        nfw_func = np.log(1+self.c)-self.c/(1+self.c)
        delta_c = 200./3*self.c**3/nfw_func
        rho_s = (delta_c*rho_c)
        l_s = 4*np.pi/3.*rho_s**2*self.r_s**3

        return ((1.-(1.+self.r200/self.r_s)**-3)*l_s /(self.coord.distance*units.kpc)**2).value