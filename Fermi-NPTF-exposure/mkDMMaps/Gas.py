import numpy as np
import healpy as hp
import mpmath as mp
from scipy import integrate
from astropy.cosmology import Planck15


class make_Gas:
    def __init__(self,z = 0.1, ell = 0, b = 0, M200 = 1.0e15,r200=0.1,nside=256,n_array = [],rc_array = [], beta_array = []):
        '''
        M200 in solar masses
        r200 in Mpc
        '''
        #variables for creating NFW template
        self.z = z
        self.ell = ell
        self.b = b
        self.nside = nside

        self.M200 = M200
        self.r200 = r200
        self.rc_array = rc_array
        self.n_array = n_array
        self.beta_array = beta_array

        self._get_values()

    def _get_values(self):
        self.distance = Planck15.comoving_distance(self.z).value #in Mpc
        self.phi = self.ell
        self.theta = np.pi/2.0 - self.b

    def get_psi(self,ell_i,b_i):
        return hp.rotator.angdist([self.theta,self.phi],[np.pi/2.0 - b_i,ell_i])
    # def get_psi(self,ell_i,b_i):
    #   # '''
    #   # return angular distance between (ell_i,b_i) and (self.ell, self.b)
    #   # '''
    #   return hp.rotator.angdist([self.theta,sef.phi],[np.pi/2.0 - b_i,ell_i])

    def _A_CR(self,r):
        '''
        from 1207.6749 Eqs. D1 - D5
        '''
        C_center = 5.0e-7
        C200 = 1.7e-7 * (self.M200/(1.0e15))**0.51
        Rtrans = 0.021*self.r200*(self.M200/(1.0e15))**0.39
        beta = 1.04 * (self.M200/(1.0e15))**0.15
        A = ((C200 - C_center)*(1+(r/Rtrans)**(-beta))**-1.0+C_center)
        return A

    def _gas_squared(self,r):
        '''
        this is rho_gas^2 from 1207.6749
        '''
        rho_gas_array = [self.n_array[i]**2.0 * (1 + (r/self.rc_array[i])**2.0 )**(-3*self.beta_array[i]) for i in range(len(self.n_array))]
        return np.sum(rho_gas_array,axis=0)


    def rho_dimless(self,r):
        '''
        this is the analgous of rho^2 but for cosmic ray emission
        '''
        #r in kpc
        return self._A_CR(r)*self._gas_squared(r)#np.power(r/self.r_s, -self.gamma_nfw) * np.power(1. + r/self.r_s, self.gamma_nfw - 3.) 

    def r_NFW(self,l, psi):
        return np.sqrt(self.distance**2. + l**2. - 2.*self.distance*l*np.cos(psi))

    def L_NFW_integral(self,ell_i,b_i):
        psi = self.get_psi(ell_i,b_i)
        return integrate.quad(lambda l: self.rho_dimless(self.r_NFW(l, psi)), self.distance - 100*np.max(self.rc_array) , self.distance + 100.*np.max(self.rc_array))[0]

    def L_NFW_integral_psi(self,psi):
        return integrate.quad(lambda l: self.rho_dimless(self.r_NFW(l, psi)),self.distance-100*np.max(self.rc_array) , self.distance + 100.*np.max(self.rc_array))[0]

    # def L_NFW_integral_theta_phi(self,theta,phi):
    #     psi = self.get_psi(ell_i,b_i)
    #     return integrate.quad(lambda l: self.rho_dimless(self.r_NFW(l, psi))**2., self.distance - 100*self.r_s, self.distance + 100.*self.r_s)[0]
 
