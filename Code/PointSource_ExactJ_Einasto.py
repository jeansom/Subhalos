import numpy as np
import scipy.integrate as integrate

from AssortedFunctions import getPPnoxsec
from ConcentrationFunc import *
from PointSource import PointSource
import units
                                                                  
alpha_Ein = 0.17

class PS(PointSource):
    def __init__(self, Mvir, Rvir, theta, phi):
        PointSource.__init__(self, Mvir, Rvir, theta, phi)

    def rho0(self, id, alpha=alpha_Ein):
        rho0 = self.Mvir[id] / integrate.quad(lambda r: 4*np.pi*r**2 * np.exp( -2/alpha * ( (r/self.r_s[id])**alpha - 1) ), 0, self.r200[id] )[0]
        return rho0
                                              
    def rho(self, rho0, r, id, alpha=alpha_Ein):
        return rho0 * np.exp( -2/alpha * ( (r/self.r_s[id])**alpha - 1) )
            
    def J_fac_integral(self):
        J_int = []
        for id, dval in enumerate(self.coord.distance.kpc):
            rho0 = self.rho0(id)
            d_halo = self.coord.distance.kpc[id]*units.kpc
            r_arr = np.linspace(1e-50, self.r200[id], 10000000)
            int_arr = 2*np.pi / d_halo**2 * self.rho(rho0, r_arr, id)**2 / ( 2 * (r_arr/d_halo) ) * np.log( ( (r_arr/d_halo) + 1)**2 / ( (r_arr/d_halo) - 1)**2 )
            J_int.append( integrate.quad(lambda r: 2*np.pi / d_halo**2 * r**2 * self.rho(rho0, r, id)**2 / ( 2 * (r/d_halo) ) * np.log( ( (r/d_halo) + 1)**2 / ( (r/d_halo) - 1)**2 ), 0, self.r200[id])[0] )
        return np.array(J_int)
