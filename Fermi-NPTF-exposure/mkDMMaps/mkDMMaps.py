import numpy as np
import scipy
import healpy as hp
from scipy import integrate, interpolate
from astropy import cosmology

from NFW import make_NFW

class mkDMMaps(make_NFW):
    def __init__(self,angle_mult=2,max_eval=100,*args,**kwargs):
        make_NFW.__init__(self,*args,**kwargs)

        #variables for making intensity map
        self.angle_mult = angle_mult
        self.npix = hp.nside2npix(self.nside)
        self.map = np.zeros(self.npix)
        self.max_eval = max_eval

        self._find_pixels()
        self._make_map()

    def _find_pixels(self):
        pix_num = hp.ang2pix(self.nside,self.theta,self.phi)
        psi_0 = np.arctan(self.r_s/self.distance)
        self.psi_0 = psi_0

        self.pixels = hp.query_disc(self.nside,hp.ang2vec(self.theta,self.phi),psi_0*self.angle_mult,inclusive=1)
        self.theta_array, self.phi_array = hp.pix2ang(self.nside,self.pixels)

    def _make_map(self):
        if len(self.pixels) < self.max_eval:
            Ls = np.vectorize(self.L_NFW_integral)(self.phi_array,np.pi/2.0-self.theta_array)
        else:
            psi_s = np.vectorize(self.get_psi)(self.phi_array,np.pi/2.0-self.theta_array)
            angs_int = np.linspace(1e-5,np.max(psi_s)*1.01,self.max_eval)
            Ls_int = np.vectorize(self.L_NFW_integral_psi)(angs_int)
            interp = interpolate.interp1d(angs_int,Ls_int)
            self.interp = interp
            self.psi_s = psi_s
            Ls = interp(psi_s)

        Ls_norm = Ls/np.sum(Ls)*self.J_0
        self.map[self.pixels] = Ls_norm

