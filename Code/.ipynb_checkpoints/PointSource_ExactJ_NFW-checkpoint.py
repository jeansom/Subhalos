import numpy as np
import scipy.integrate as integrate
from PointSource import PointSource
import units                               

# Parameters for MW NFW profile                                                                      
gamma_NFW = 1

class PS(PointSource):
    def __init__(self, Mvir, Rvir, theta, phi, rsatfac):
        PointSource.__init__(self, Mvir, Rvir, theta, phi)
        self.rsatfac = rsatfac
        self.rsat = self.rsatfac*self.r200

    def rho0(self, id, gamma=gamma_NFW):
        rho0 = self.Mvir[id] / (4*np.pi*self.r_s[id]**3*( np.log(1 + self.r200[id]/self.r_s[id]) - self.r200[id]/(self.r200[id]+self.r_s[id]) ))
        return rho0
                                              
    def rho(self, rho0, r, id, gamma=gamma_NFW):
        if r > self.rsat[id]:
            return rho0/( (r/self.r_s[id])**gamma * (1+r/self.r_s[id])**(3-gamma) )
        else:
            return (4*np.pi*rho0*self.r_s[id]**3*( np.log(1 + self.rsat[id]/self.r_s[id]) - self.rsat[id]/(self.rsat[id]+self.r_s[id]) ))/(4/3 * np.pi * self.rsat[id]**3 )

    def J_fac_integral(self):
        J_int = []
        for id, dval in enumerate(self.coord.distance.kpc):
            rho0 = self.rho0(id)
            d_halo = self.coord.distance.kpc[id]*units.kpc
            l_halo = self.coord.l.radian[id]
            d = np.linspace(0, dval*units.kpc + self.r200[id], 20000000)
            theta = np.linspace(0, np.pi, 10000)
            J_int.append( integrate.quad(lambda d: self.rho(rho0, self.r_halocentric(d_halo, d), id)**2, max(0,dval*units.kpc-10*self.r200[id]), dval*units.kpc + 10*self.r200[id])[0] )
        return np.array(J_int)