import numpy as np
import healpy as hp
import mpmath as mp
from scipy import integrate
from astropy.cosmology import Planck15


class make_NFW:
    def __init__(self,z = 0.1,r_s = 0.02, J_0 = 1, ell = 0, b = 0, gamma_nfw = 1.00,nside=256,use_boost=False, Burkert=False):
        #variables for creating NFW template
        self.z = z
        self.r_s = r_s #in Mpc
        self.J_0 = J_0
        self.ell = ell
        self.b = b
        self.gamma_nfw = gamma_nfw
        self.nside = nside
        self.use_boost = use_boost
        self.Burkert = Burkert

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

    def rho_dimless(self,r):
        # Return NFW unless ask for Burkert
        #r in kpc
        if self.Burkert:
            rB = 0.666511 * self.r_s
            return 1./( (1. + r/rB) * (1. + (r/rB)**2) )
        else:
            return np.power(r/self.r_s, -self.gamma_nfw) * np.power(1. + r/self.r_s, self.gamma_nfw - 3.) 

    def r_NFW(self,l, psi):
        return np.sqrt(self.distance**2. + l**2. - 2.*self.distance*l*np.cos(psi))

    def L_NFW_integral(self,ell_i,b_i):
        psi = self.get_psi(ell_i,b_i)
        if not self.use_boost:
            return integrate.quad(lambda l: self.rho_dimless(self.r_NFW(l, psi))**2., self.distance - 100*self.r_s, self.distance + 100.*self.r_s)[0] ##Return NFW^2
        else:
            return integrate.quad(lambda l: self.rho_dimless(self.r_NFW(l, psi)), self.distance - 100*self.r_s, self.distance + 100.*self.r_s)[0] ##Return NFW


    def L_NFW_integral_psi(self,psi):
        if not self.use_boost:
            return integrate.quad(lambda l: self.rho_dimless(self.r_NFW(l, psi))**2., self.distance - 100*self.r_s, self.distance + 100.*self.r_s)[0]
        else:
            return integrate.quad(lambda l: self.rho_dimless(self.r_NFW(l, psi)), self.distance - 100*self.r_s, self.distance + 100.*self.r_s)[0]

    # def L_NFW_integral_theta_phi(self,theta,phi):
    #     psi = self.get_psi(ell_i,b_i)
    #     return integrate.quad(lambda l: self.rho_dimless(self.r_NFW(l, psi))**2., self.distance - 100*self.r_s, self.distance + 100.*self.r_s)[0]
 
