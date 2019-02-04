"""
Module for calculating the source counts and intensities of 
astrophysical objects (blazars, SFGs, mAGNs).

Created by Siddharth Mishra-Sharma. Last modified 05/23/2016
"""

import sys, os
import collections
import functools
import random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as ip
from scipy import integrate
import pandas as pd
import healpy as hp
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy.special import hyp2f1
import vegas # Monte-carlo integration modules

from constants import *
import CosmologicalDistance as cld

class LuminosityFunctionBL:
    """
    Class for calculating the luminosity function, source count function and intensity spectrum of blazars (BL Lacs and FSRQs). 
    Includes LFs from 1110.3787 (FSRQ), 1310.0006 (BL Lacs) and 1501.05301 (BL Lacs + FSRQs) (all Ajello et al).
    """
    def __init__(self, model = 'blazars', sed = 'ajello', ebl = True, lcut = False, data_dir = '../data'):

        # Set options
        self.model = model # Which LF model to use (see models below)
        self.sed = sed # Which SED to use -- a power law ('pl') or the double power law from Ajello et al ('ajello')
        self.ebl = ebl # Whether to use EBL suppression (takes longer)
        self.lcut = lcut # Whether to have a lower cutoff on luminosity
        self.data_dir = data_dir # Where the data is stored

        self.initialize_LF_models()
        self.initialize_int_ranges()

        # Energy range over which LF is defined
        self.E10 = 100*MeV
        self.E20 = 100*GeV


        self.cd = cld.CosmologicalDistance() # For cosmological calculations. Default flat Lambda CDM parameters H0 = 70, \Omega_m = 0.3, \Omega_\Lambda = 0.7
        self.make_Tau() # Create EBL interpolation based on Finke et al


    def initialize_LF_models(self):
        """ Which LF model to initialize """
        if self.model == 'blazars':
            self.params_BLL_FSRQ() # Blazars + FSRQs (1501.05301)
        elif self.model == 'fsrq':
            self.params_FSRQ() # FSRQs (1110.3787)
        elif self.model == 'bll1':
            self.params_BLLacs1() # BL Lac LDDE model 1 (1310.0006)
        elif self.model == 'bll2':
            self.params_BLLacs2() # BL Lac LDDE model 2 (1310.0006)
        elif self.model == 'bllhsp':
            self.params_BLLacsHSP() # BL Lac LDDE HSP (1310.0006)
        elif self.model == 'bllisplsp':
            self.params_BLLacsISPLSP() # BL Lac LDDE ISP+LSP (1310.0006)

    def initialize_int_ranges(self):
        """ Integration ranges """
        if self.model == 'fsrq':
            # Integration ranges, modified from equation (5) of 1110.3787
            self.Lmin = 10**44*erg*Sec**-1
            self.Lmax = 10**51*erg*Sec**-1 #Upper limit taken down slightly to help with integration convergence
            self.zmin = 10**-3
            self.zmax = 5. #Upper limit taken down slightly because that's up to where the Finke et al EBL files go
            self.Gammamin = 1.8
            self.Gammamax = 3.0
        elif self.model == 'blazars':
            # Integration ranges, modified from equation (12) of 1501.05301
            self.Lmin = 10**43*erg*Sec**-1
            self.Lmax = 10**51*erg*Sec**-1 #Upper limit taken down slightly to help with integration convergence
            self.zmin = 10**-3
            self.zmax = 5. #Upper limit taken down slightly because that's up to where the Finke et al EBL files go
            self.Gammamin = 1.0
            self.Gammamax = 3.5
        else:
            # Integration ranges, modified from equation (4) of 1310.0006
            self.Lmin = 7*10**43*erg*Sec**-1
            self.Lmax = 10**51*erg*Sec**-1 #Upper limit taken down slightly to help with integration convergence
            self.zmin = 0.03
            self.zmax = 5. #Upper limit taken down slightly because that's up to where the Finke et al EBL files go
            self.Gammamin = 1.45
            self.Gammamax = 2.8

    """
    ************************************************************
    Parameters
    """

    def params_BLL_FSRQ(self):
        """
        Parameters for blazars (BL Lacs + FSRQs), taken from Table 1 of 1501.05301 (model 1)
        assuming Luminosity-Dependent Density Evolution (LDDE)
        """
        self.A = 196*10**-2*(1000*Mpc)**-3*erg**-1*Sec
        self.gamma1 = 0.50
        self.Lstar = (1.05)*10**48*erg*Sec**-1
        self.gamma2 = 1.83
        self.zcstar = 1.25
        self.p1star = 3.39
        self.tau = 3.16
        self.p2star = -4.96
        self.alpha = 7.23*10**-2
        self.mustar = 2.22
        self.beta = 0.10
        self.sigma = 0.28
        self.delta = 0.64

    def params_FSRQ(self):
        """
        Parameters for FSRQs, taken from Table 3 of 1110.3787
        assuming Luminosity Dependent Density Evolution (LDDE)
        """
        self.A = (3.06*10**4)*(10**-13*Mpc**-3*erg**-1*Sec)
        self.gamma1 = 0.21
        self.Lstar = (.84)*10**48*erg*Sec**-1
        self.gamma2 = 1.58
        self.zcstar = 1.47
        self.p1star = 7.35
        self.tau = 0.
        self.p2star = -6.51
        self.alpha = 0.21
        self.mustar = 2.44
        self.beta = 0.
        self.sigma = 0.18
        self.delta = 0

    def params_BLLacs1(self):
        """
        Parameters for BL Lacs, taken from Table 3 of 1301.0006 (model 1)
        assuming Luminosity Dependent Density Evolution (LDDE)
        """
        self.A = (9.20*10**2)*(10**-13*Mpc**-3*erg**-1*Sec)
        self.gamma1 = 1.12
        self.Lstar = (2.43)*10**48*erg*Sec**-1
        self.gamma2 = 3.71
        self.zcstar = 1.67
        self.p1star = 4.50
        self.tau = 0.
        self.p2star = -12.88
        self.alpha = 4.46*10**-2
        self.mustar = 2.12
        self.beta = 6.04*10**-2
        self.sigma = 0.26
        self.delta = 0

    def params_BLLacs2(self):
        """
        Parameters for BL Lacs, taken from Table 3 of 1301.0006 (model 2)
        assuming Luminosity Dependent Density Evolution (LDDE)
        """
        self.A = (3.39*10**4)*(10**-13*Mpc**-3*erg**-1*Sec)
        self.gamma1 = 0.27
        self.Lstar = (.28)*10**48*erg*Sec**-1
        self.gamma2 = 1.86
        self.zcstar = 1.34
        self.p1star = 2.24
        self.tau = 4.92
        self.p2star = -7.37
        self.alpha=4.53*10**-2
        self.mustar = 2.10
        self.beta = 6.46*10**-2
        self.sigma = 0.26
        self.delta = 0

    def params_BLLacsHSP(self):
        """
        Parameters for BL Lacs, taken from Table 7 of 1301.0006 (model HSP)
        assuming Luminosity Dependent Density Evolution (LDDE)
        """
        self.A = (9.59)*(10**-10*Mpc**-3*erg**-1*Sec)
        self.gamma1 = 0.28
        self.Lstar = (.42)*10**48*erg*Sec**-1
        self.gamma2 = 3.47
        self.zcstar = 1.60
        self.p1star = 0.48
        self.tau = 6.76
        self.p2star = -11.12
        self.alpha=0.11
        self.mustar = 1.97
        self.beta = 4.40*10**-2
        self.sigma = 0.24
        self.delta = 0

    def params_BLLacsISPLSP(self):
        """
        Parameters for BL Lacs, taken from Table 7 of 1301.0006 (model ISP+LSP)
        assuming Luminosity Dependent Density Evolution (LDDE)
        """
        self.A = (17.1)*(10**-10*Mpc**-3*erg**-1*Sec)
        self.gamma1 = 0.48
        self.Lstar = (.45)*10**48*erg*Sec**-1
        self.gamma2 = 1.98
        self.zcstar = 1.15
        self.p1star = 4.54
        self.tau = 3.82
        self.p2star = -5.89
        self.alpha=4.69*10**-3
        self.mustar = 2.26
        self.beta = -2.81*10**-2
        self.sigma = 0.20
        self.delta = 0

    """
    ************************************************************
    """

    # @memoized
    def phi_LDDE(self, L, z, Gamma):
        """
        Parameterization of Luminosity-Dependent Density Evolution (LDDE) model. Equations (10)-(20) of Ajello et al.
        Returns Phi(L,V,Gamma) = d^3N/(dLdVdGamma). Note the formula for e(L,z) has a sign error in 1110.3787 and 1310.0006
        """
        self.zc = self.zcstar*(L/(10**48*erg*Sec**-1))**self.alpha
        self.p1 = self.p1star+self.tau*(np.log10(L/(erg*Sec**-1)) - 46)
        self.p2 = self.p2star+self.delta*(np.log10(L/(erg*Sec**-1)) - 46)
        self.e = (((1+z)/(1+self.zc))**-self.p1+((1+z)/(1+self.zc))**-self.p2)**-1
        self.mu = self.mustar + self.beta*(np.log10(L/(erg*Sec**-1))-46)
        
        self.phi = (self.A/(np.log(10)*L/(erg*Sec**-1)))*((L/self.Lstar)**self.gamma1+(L/self.Lstar)**self.gamma2)**-1*self.e*np.exp(-(Gamma - self.mu)**2/(2*self.sigma**2))
        return self.phi

    def make_Tau(self):
        """
        Create EBL interpolation (model based on Finke et al)
        """
        # Load and combine EBL files downloaded from http://www.phy.ohiou.edu/~finke/EBL/ 
        tau_files = self.data_dir + '/tau_modelC_total/tau_modelC_total_z%.2f.dat'
        z_list = np.arange(0, 5, 0.01) # z values to load files for
        E_list, tau_list = [],[]
        for z in z_list:
            d = np.genfromtxt(tau_files % z, unpack=True)
            E_list = d[0]
            tau_list.append(d[1])
        self.Tau_ip = ip.RectBivariateSpline(z_list, np.log10(E_list), np.log10(np.array(tau_list))) # Create interpolation
    
    def Tau(self,E,z):
        """
        EBL attenuation of gamma rays (exponential factor)
        """
        return np.float64(10**self.Tau_ip(z, np.log10(E/(1000*GeV)))) # Model of Finke et al
        # return (z/3.3)*(E/(10*GeV))**.8 # Analytic approximation from 1506.05118

    def dFdE(self, E,z, L, Gamma):
        """
        Intrinsic flux of source for calculating spectrum
        """
        if self.sed == "pl":

            dL = self.cd.luminosity_distance(z)*Mpc
            Kcorr = (1+z)**(2-Gamma)
            N = L/(4*np.pi*dL**2)/((1/self.E10)**-Gamma*(self.E20**(-Gamma+2)-self.E10**(-Gamma+2))/(-Gamma+2)) # Check it double deck it 
            
            return N*Kcorr*((E/self.E10)**-Gamma) # Does not include EBL suppression
        
        else:
            gammaa=1.7
            gammab=2.6

            dL = self.cd.luminosity_distance(z)*Mpc
            Kcorr = (1+z)**(2-Gamma)
            N = L/(4*np.pi*dL**2)/(0.00166667*self.Eb(Gamma)**2.6*(hyp2f1(0.666667, 1., 1.66667, -0.0000316228*self.Eb(Gamma)**0.9) - 0.0158489*hyp2f1(0.666667, 1.,1.66667, -6.30957*10**-8*self.Eb(Gamma)**0.9)))
            
            return N*Kcorr*((E/self.Eb(Gamma))**gammaa+(E/self.Eb(Gamma))**gammab)**-1 # Does not include EBL suppression

    def Eb(self, Gamma):
        """
        From Ajello et al (text in page 7)
        """
        return 10**(9.25 - 4.11*Gamma)*GeV
    
    """
    ************************************************************
    Monte carlo integration experimentation
    """

    def dIdE_integrand(self, x):
        
        Gamma = x[0]
        L = x[1]
        z = x[2]
        return self.dVdz(z)*self.phi_LDDE(L,z, Gamma)*self.dFdE(self.E,z,L, Gamma)*np.exp(-self.Tau(self.E,z))

    def dIdE_mc_vegas(self, E, nitn=15, neval=2e4, verbose=False):
        self.E = E

        integ = vegas.Integrator([[self.Gammamin,self.Gammamax], [self.Lmin,self.Lmax], [self.zmin,self.zmax]])
        result = integ(self.dIdE_integrand, nitn=nitn, neval=neval)
        if verbose:
            print(result.summary())
        return result.mean

    """
    ************************************************************
    """
    
    def dIdE(self, E):
        """
        Return intensity spectrum of blazars. Since this is only used for sub-bin apportioning of photons, 
        we use a single index approximation (the source count function uses the full form)
        """

        Gamma = self.mustar # Assumed spectral index for the class

        self.dIdEval = integrate.nquad(lambda L,z: self.dVdz(z)*self.phi_LDDE(L,z, Gamma)*self.dFdE(E,z,L, Gamma)*np.exp(-self.Tau(E,z)),[[self.Lmin,self.Lmax], [self.zmin, self.zmax]], opts=[{'epsrel':1e-2,'epsabs':0},{'epsrel':1e-2,'epsabs':0}])[0]

        return self.dIdEval

    def dVdz(self,z):
        """
        Return comoving volument element
        """
        return self.cd.comoving_volume_element(z)*Mpc**3

    def Lgamma(self, Fgamma, Gamma,z):
        """
        Return luminosity flux given energy flux Fgamma
        """
        dL = self.cd.luminosity_distance(z)*Mpc
        if self.sed == "pl":
            return 4*np.pi*dL**2*(1+z)**(-2+Gamma)*Fgamma*((self.E20**(-Gamma+2)-self.E10**(-Gamma+2))/(self.E20**(-Gamma+1)-self.E10**(-Gamma+1)))*((-Gamma+1)/(-Gamma+2))
        else:
            return 4*np.pi*dL**2*(1+z)**(-2+Gamma)*Fgamma*(0.00166667*self.Eb(Gamma)**2.6*(1.*hyp2f1(0.666667, 1., 1.66667, -0.0000316228*self.Eb(Gamma)**0.9) - 0.0158489*hyp2f1(0.666667, 1., 1.66667, -6.30957*10**-8*self.Eb(Gamma)**0.9)))/(6.25*10**-9*self.Eb(Gamma)**2.6*(1.*hyp2f1(1., 1.77778, 2.77778, -0.0000316228*self.Eb(Gamma)**0.9) - 0.0000158489*hyp2f1(1., 1.77778, 2.77778, -6.30957*10**-8*self.Eb(Gamma)**0.9)))

    def step(self, x):
        """
        Step function if this is enabled, otherwise does nothing
        """
        if self.lcut:
            return 1.*(x > 0)
        else:
            return 1.


    def dNdF(self,Fgamma):
        """
        Returns the differential source counts function
        """    
        dFgamma = Fgamma/1000
        return (1/dFgamma)*integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma_var: self.step(self.Lgamma(Fgamma, Gamma, z) - self.Lmin)*4*np.pi*self.dVdz(z)*self.phi_LDDE(Lgamma_var,z, Gamma), self.Lgamma(Fgamma , Gamma,z), self.Lgamma(Fgamma+dFgamma, Gamma,z))[0],self.zmin,self.zmax)[0],self.Gammamin,self.Gammamax)[0]

    def set_dIdE(self, Evals, dIdEvals):
        """
        Make interpolating function from calculated energy spectrum
        """
        self.dIdE_interp = ip.InterpolatedUnivariateSpline(Evals, dIdEvals)

    def Fpgamma_EBL(self,Fgamma,E1,E2, Gamma):
        """
        Stretch flux for a given range to observed value over 0.1-100 GeV, with EBL correction. To use as an interpolation in integrals (otherwise impossibly slow).
        """
        self.E10 = 100*MeV
        self.E20 = 100*GeV

        numerator = integrate.nquad(lambda z,E: self.dFdE(E, z, self.Lgamma(Fgamma, Gamma, z), Gamma)*np.exp(-self.Tau(E,z)), [[self.zmin,self.zmax] ,[self.E10, self.E20]])[0]
        denominator = integrate.nquad(lambda z,E: self.dFdE(E, z, self.Lgamma(Fgamma, Gamma, z), Gamma)*np.exp(-self.Tau(E,z)), [[self.zmin,self.zmax] ,[E1, E2]])[0]
        
        return Fgamma*numerator/denominator

    def make_Fpgamma(self, Fgamma, E1, E2):
        """
        Create Fpgamma interpolation with EBL correction
        """
        Gamma_vals = np.linspace(self.Gammamin, self.Gammamax, 40)
        Fpgamma_vals = [self.Fpgamma_EBL(Fgamma, E1, E2, Gamma) for Gamma in Gamma_vals]

        Fpgamma_interp = ip.InterpolatedUnivariateSpline(Gamma_vals, Fpgamma_vals)

        return Fpgamma_interp

    def Fpgamma(self,Fgamma,E1,E2, Gamma):
        """
        Stretch flux for a given range to observed value over 0.1-100 GeV, analytic form
        """

        if self.ebl:
            return np.float64(self.Fpgamma_interp(Gamma))
        else:
            if self.sed == "pl":
                return Fgamma*((self.E20**(-Gamma+1)-self.E10**(-Gamma+1))/(E2**(-Gamma+1)-E1**(-Gamma+1)))
            else:
                return Fgamma*(6.25*10**-9*self.Eb(Gamma)**2.6*(1.*hyp2f1(1., 1.77778, 2.77778, -0.0000316228*self.Eb(Gamma)**0.9) - 0.0000158489*hyp2f1(1., 1.77778, 2.77778, -6.30957*10**-8*self.Eb(Gamma)**0.9)))/((0.625*E2**1.6*hyp2f1(1., 1.77778, 2.77778, -(1/(E1/self.Eb(Gamma))**0.9)) - 0.625*E1**1.6*hyp2f1(1., 1.77778, 2.77778, -(1/(E2/self.Eb(Gamma))**0.9)))/((E1*E2)**1.6*(1/self.Eb(Gamma))**2.6))
        
    def dNdFp(self,Fgamma,E1,E2):
        """
        Returns the differential source counts function in units of Centimeter**2*Sec
        """    
        
        dFgamma = Fgamma/1000

        if self.ebl:    
            self.Fpgamma_interp_low = self.make_Fpgamma(Fgamma, E1, E2)
            self.Fpgamma_interp_high = self.make_Fpgamma(Fgamma+dFgamma, E1, E2)
            return (1/dFgamma)*integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma_var: self.step(self.Lgamma(Fgamma, Gamma, z) - self.Lmin)*4*np.pi*self.dVdz(z)*self.phi_LDDE(Lgamma_var,z, Gamma), self.Lgamma(self.Fpgamma_interp_low(Gamma) , Gamma,z), self.Lgamma(self.Fpgamma_interp_high(Gamma), Gamma,z))[0],self.zmin,self.zmax)[0],self.Gammamin,self.Gammamax,epsabs=0,epsrel=10**-2)[0]
        else:
            return (1/dFgamma)*integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma_var: self.step(self.Lgamma(Fgamma, Gamma, z) - self.Lmin)*4*np.pi*self.dVdz(z)*self.phi_LDDE(Lgamma_var,z, Gamma), self.Lgamma(self.Fpgamma(Fgamma,E1,E2, Gamma) , Gamma,z), self.Lgamma(self.Fpgamma(Fgamma+dFgamma,E1,E2,Gamma), Gamma,z))[0],self.zmin,self.zmax)[0],self.Gammamin,self.Gammamax,epsabs=0,epsrel=10**-2)[0]

class LuminosityFunctionSFG:
    """
    Class for calculating the luminosity function and source counts of SFGs.
    Based on 1404.1189, 1302.5209 and 1206.1346.
    """
    def __init__(self, source, pionic_peak = True, data_dir = ''):
        
        self.source = source # Set the source type
        self.pionic_peak = pionic_peak # Do we want the pionic peak energy spectrum

        # Set a few general parameters
        self.alpha = 1.17 # These are values from 1206.1346 (also given after eq. 2.4 of 1404.1189)
        self.beta = 39.28 
        self.Ls = 3.828*10**33*erg*Sec**-1 # Verify this is the right value for the solar luminosity -- from https://en.wikipedia.org/wiki/Solar_luminosity
        
        self.set_params() # Set the LF parameters

        # Integration ranges
        self.Lmin = 10**33*erg*Sec**-1 
        self.Lmax = 10**44*erg*Sec**-1
        self.zmin = 0.
        self.zmax = 4.19

        self.CTB_en_bins = 10**np.linspace(np.log10(0.3), np.log10(300),31) # CTB energy bins

        self.cd = cld.CosmologicalDistance() # For cosmological distance calculations

        self.make_Tau() # Create EBL interpolation from Finke et al

    def set_params(self):
        """
        Parameters from Table 8 of 1302.5209
        """    
        if self.source == 'NG':

            self.phi_IR_star_0 = 10**-1.95*Mpc**-3
            self.LIR_star_0 = 10**9.45*self.Ls

            self.alpha_IR = 1.00
            self.sigma_IR = 0.5

            self.kL1 = 4.49
            self.kL2 = 0.00
            self.zbL = 1.1

            self.kRh1 = -0.54
            self.kRh2 = -7.13
            self.zbRh = 0.53

            self.Gamma = 2.7

        elif self.source == 'SB':

            self.phi_IR_star_0 = 10**-4.59*Mpc**-3
            self.LIR_star_0 = 10**11.0*self.Ls

            self.alpha_IR = 1.00
            self.sigma_IR = 0.35

            self.kL1 = 1.96

            self.kRh1 = 3.79
            self.kRh2 = -1.06
            self.zbRh = 1.1

            self.Gamma = 2.2

        elif self.source == 'SF-AGN':

            self.phi_IR_star_0 = 10**-3.00*Mpc**-3
            self.LIR_star_0 = 10**10.6*self.Ls

            self.alpha_IR = 1.20
            self.sigma_IR = 0.4

            self.kL1 = 3.17

            self.kRh1 = 0.67
            self.kRh2 = -3.17
            self.zbRh = 1.1

            self.GammaSB = 2.2
            self.GammaNG = 2.7

        elif self.source == 'ALL':

            self.phi_IR_star_0 = 10**-2.29*Mpc**-3
            self.LIR_star_0 = 10**10.12*self.Ls

            self.alpha_IR = 1.15
            self.sigma_IR = 0.52

            self.kL1 = 3.55
            self.kL2 = 1.62
            self.zbL = 1.85

            self.kRh1 = -0.57
            self.kRh2 = -3.92
            self.zbRh = 1.1

            self.Gamma = 2.475

    def phi_IR_star(self, z):
        """
        Redshift evolution of phi_IR (see section 3.5 of 1302.5209)
        """
        if z < self.zbRh:
            return self.phi_IR_star_0*(1+z)**self.kRh1
        else:
            return (self.phi_IR_star_0*(1+z)**self.kRh2)*(self.phi_IR_star_0*(1+self.zbRh)**self.kRh1)/((self.phi_IR_star_0*(1+self.zbRh)**self.kRh2))

    def LIR_star(self, z):
        """
        Redshift evolution of LIR (see section 3.5 of 1302.5209)
        """

        if (self.source == 'NG') | (self.source == 'ALL'):
            if z < self.zbL:
                return self.LIR_star_0*(1+z)**self.kL1
            else:
                return (self.LIR_star_0*(1+z)**self.kL2)*(self.LIR_star_0*(1+self.zbL)**self.kL1)/(self.LIR_star_0*(1+self.zbL)**self.kL2)
        else:
            return self.LIR_star_0*(1+z)**self.kL1

    def phi_IR(self, LIR, z):
        """
        Returns the IR luminosity for a give SFG sub-class.
        Based on eq. 2.2 of 1404.1189.
        """
        return self.phi_IR_star(z)*(LIR/self.LIR_star(z))**(1-self.alpha_IR)*np.exp(-(1/(2*self.sigma_IR**2))*(np.log10(1+(LIR/self.LIR_star(z))))**2)


    def LIR(self, Lgamma):
        """
        Returns the IR luminosity given the gamma luminosity.
        Based on eq. 2.4 of 1404.1189.
        """
        return ((Lgamma/(erg*Sec**-1))*(1/10**self.beta))**(1/self.alpha)*10**10*self.Ls

    def phi_gamma(self, Lgamma, z):
        """
        Returns the gamma-ray LF given the IR LF.
        Based on eq. 2.5 of 1404.1189.
        This is defined as Phi(Lgamma,z)= dN/dVdLog(Lgamma) according to text after eq. 2.1 so we divide by ln(10)*Lgamma to get standard form dN/dVdLgamma we use as per usual.
        """
        return self.phi_IR(self.LIR(Lgamma),z)*(1/self.alpha)*(1/(np.log(10)*Lgamma))

    def dFdE_unnorm(self, E, Gamma):
        if self.pionic_peak:
            if E < 600*MeV:
                return E**-1.5
            else:
                return E**-Gamma    
        else:
            return E**-Gamma

    def Lgamma(self, Fgamma, Gamma,z):
        """
        Return luminosity flux given energy flux Fgamma
        """
        dL = self.cd.luminosity_distance(z)*Mpc
        E1=100*MeV
        E2=100*GeV
        if self.pionic_peak:
            N = Fgamma/(((600*MeV)**-0.5 - (100*MeV)**-0.5)/(-0.5*(600*MeV)**-1.5)*(1+z)**-1.5 + ((100*GeV)**(-Gamma+1) - (600*MeV)**(-Gamma+1))/((-Gamma+1)*(600*MeV)**-Gamma)*(1+z)**-Gamma) # Normalization for BPL form in Tamborra et al
            L_N = (4*np.pi*dL**2)*(((600*MeV)**0.5 - (100*MeV)**0.5)/(0.5*(600*MeV)**-1.5)*(1+z)**-1.5 + ((100*GeV)**(-Gamma+2) - (600*MeV)**(-Gamma+2))/((-Gamma+2)*(600*MeV)**-Gamma)*(1+z)**-Gamma)
        else:
            N = Fgamma/(E2**-(Gamma+1)-E1**-(Gamma+1)) # Normalization for PL # This is wrong -- check
            L_N = (4*np.pi*dL**2)*(E2**-(Gamma+2)-E1**-(Gamma+2)) # This is wrong -- check

        return N*L_N

    def fSB(self, z):
        """
        Fraction of SF-AGN sources contributing SB and non-SB type 
        spectra, Table 2 of Tamborra et al
        """
        if self.source == 'SF-AGN':
            if 0.0 <= z < 0.3:
                return .15
            elif 0.3 <= z < 0.45:
                return .09
            elif 0.45 <= z < 0.6:
                return .01
            elif 0.6 <= z < 0.8:
                return .13
            elif 0.8 <= z < 1.0:
                return .27
            elif 1.0 <= z < 1.2:
                return .68
            elif 1.2 <= z < 1.7:
                return .25
            elif 1.7 <= z < 2.0:
                return .25
            elif 2.0 <= z < 2.5:
                return .81
            elif 2.5 <= z < 3.0:
                return .76
            elif 3.0 <= z < 4.2:
                return .72

    def dNdF(self, Fgamma):
        """
        Returns the differential source counts function in units of Centimeter**2*Sec
        """    
        dFgamma = Fgamma/1000

        if self.source == 'SF-AGN':
            return (1/dFgamma)*(integrate.quad(lambda z: integrate.quad(lambda Lgamma: self.fSB(z)*4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fgamma,self.GammaSB,z), self.Lgamma(Fgamma+dFgamma,self.GammaSB,z))[0], self.zmin,self.zmax)[0]+integrate.quad(lambda z: integrate.quad(lambda Lgamma: (1-self.fSB(z))*4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fgamma,self.GammaNG,z), self.Lgamma(Fgamma+dFgamma,self.GammaNG,z))[0], self.zmin,self.zmax)[0])
        else:
            return (1/dFgamma)*integrate.quad(lambda z: integrate.quad(lambda Lgamma: 4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fgamma,self.Gamma,z), self.Lgamma(Fgamma+dFgamma,self.Gamma,z))[0], self.zmin,self.zmax)[0]

    def Fpgamma(self,Fgamma,E1,E2):
        """
        Stretch flux for a given range to observed value over 0.1-100 GeV
        """
        return Fgamma*integrate.quad(lambda E: self.dIdE_interp(E), 100*MeV, 100*GeV)[0]/integrate.quad(lambda E: self.dIdE_interp(E), E1, E2)[0]

    def dNdFp(self,Fgamma,E1,E2):
        """
        Returns the scaled differential source counts function in units of Centimeter**2*Sec
        """    
        dFgamma = Fgamma/1000
        Fpgamma_L = self.Fpgamma(Fgamma,E1,E2)
        Fpgamma_H = self.Fpgamma(Fgamma+dFgamma,E1,E2)

        if self.source == 'SF-AGN':
            return (1/dFgamma)*(integrate.quad(lambda z: integrate.quad(lambda Lgamma: self.fSB(z)*4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fpgamma_L,self.GammaSB,z), self.Lgamma(Fpgamma_H,self.GammaSB,z),epsabs=0,epsrel=10**-2)[0], self.zmin,self.zmax,epsabs=0,epsrel=10**-2)[0]+integrate.quad(lambda z: integrate.quad(lambda Lgamma: (1-self.fSB(z))*4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fpgamma_L,self.GammaNG,z), self.Lgamma(Fpgamma_H,self.GammaNG,z),epsabs=0,epsrel=10**-2)[0], self.zmin,self.zmax,epsabs=0,epsrel=10**-2)[0])
        else:
            return (1/dFgamma)*integrate.quad(lambda z: integrate.quad(lambda Lgamma: 4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fpgamma_L,self.Gamma,z), self.Lgamma(Fpgamma_H,self.Gamma,z),epsabs=0,epsrel=10**-2)[0], self.zmin,self.zmax,epsabs=0,epsrel=10**-2)[0]
    
    def opts0(self,*args, **kwargs):
            return {'epsrel':1e-2,'epsabs':0}

    def dIdz(self, z, E1,E2):
        if self.source == 'SF-AGN':
            self.dIdzval = integrate.quad(lambda E: self.fSB(z)*self.dVdz(z)*integrate.quad(lambda L: self.phi_gamma(L,z)*self.dFdE((1+z)*E,z,L, self.GammaSB), self.Lmin,self.Lmax,epsabs=0,epsrel=10**-2)[0], E1, E2,epsabs=0,epsrel=10**-2)[0]+integrate.quad(lambda E: (1-self.fSB(z))*self.dVdz(z)*integrate.quad(lambda L: self.phi_gamma(L,z)*self.dFdE((1+z)*E,z,L, self.GammaNG), self.Lmin,self.Lmax,epsabs=0,epsrel=10**-2)[0], E1, E2,epsabs=0,epsrel=10**-2)[0]
        else:
            self.dIdzval = integrate.quad(lambda E: self.dVdz(z)*integrate.quad(lambda L: self.phi_gamma(L,z)*self.dFdE((1+z)*E,z,L, self.Gamma), self.Lmin,self.Lmax,epsabs=0,epsrel=10**-2)[0], E1, E2,epsabs=0,epsrel=10**-2)[0]

        return self.dIdzval
    def dFdE(self, E,z, L, Gamma):
        """
        Intrinsic flux of source. Use simple power law.
        """
        E1 = 100*MeV
        E2 = 100*GeV
        dL = self.cd.luminosity_distance(z)*Mpc
        if self.pionic_peak:
            Gamma_L = 1.5
            N = L/(4*np.pi*dL**2)/((((600*MeV)**(-Gamma_L + 2) - (100*MeV)**(-Gamma_L + 2))/((-Gamma_L+2)*(600*MeV)**-Gamma_L))*(1+z)**-Gamma_L+(((100*GeV)**(-Gamma + 2)-(600*MeV)**(-Gamma + 2))/((-Gamma+2)*(600*MeV)**-Gamma))*(1+z)**-Gamma)
            if E < 600*MeV:
                return N*(E**-Gamma_L)/(600*MeV)**-Gamma_L
            else:
                return N*(E**-Gamma)/(600*MeV)**-Gamma

        else:
            N = L/(4*np.pi*dL**2)*(-Gamma+2)/(E2**(-Gamma+2)-E1**(-Gamma+2)) # This is wrong I think -- change.
            return N*E**-Gamma

    def dIdE(self, E):
        if self.source == 'SF-AGN':
            self.dIdEval = integrate.quad(lambda z: self.fSB(z)*self.dVdz(z)*integrate.quad(lambda L: self.phi_gamma(L,z)*self.dFdE((1+z)*E,z,L, self.GammaSB), self.Lmin,self.Lmax,epsabs=0,epsrel=10**-2)[0], self.zmin, self.zmax,epsabs=0,epsrel=10**-2)[0]+integrate.quad(lambda z: (1-self.fSB(z))*self.dVdz(z)*integrate.quad(lambda L: self.phi_gamma(L,z)*self.dFdE((1+z)*E,z,L, self.GammaNG), self.Lmin,self.Lmax,epsabs=0,epsrel=10**-2)[0], self.zmin, self.zmax,epsabs=0,epsrel=10**-2)[0]
        else:
            self.dIdEval = integrate.quad(lambda z: self.dVdz(z)*integrate.quad(lambda L: self.phi_gamma(L,z)*self.dFdE((1+z)*E,z,L, self.Gamma), self.Lmin,self.Lmax,epsabs=0,epsrel=10**-2)[0], self.zmin, self.zmax,epsabs=0,epsrel=10**-2)[0]

        return self.dIdEval

    def set_dIdE(self, Evals, dIdEvals):
        """
        Make interpolating function from calculated energy spectrum
        """
        self.dIdE_interp = ip.InterpolatedUnivariateSpline(Evals, dIdEvals)

    def dVdz(self,z):
        """
        Return comoving volument element
        """
        return self.cd.comoving_volume_element(z)*Mpc**3

    def make_Tau(self):
        """
        Create EBL interpolation (model based on Finke et al)
        """

        # Load and combined EBL files downloaded from http://www.phy.ohiou.edu/~finke/EBL/ 
        tau_files = self.data_dir + '/tau_modelC_total/tau_modelC_total_z%.2f.dat'
        z_list = np.arange(0, 5, 0.01) # z values to load files for
        E_list, tau_list = [],[]
        for z in z_list:
            d = np.genfromtxt(tau_files % z, unpack=True)
            E_list = d[0]
            tau_list.append(d[1])
        self.Tau_ip = ip.RectBivariateSpline(z_list, E_list, np.array(tau_list)) # Create interpolation
    
    def Tau(self,E,z):
        """
        EBL attenuation of gamma rays using Finke et al
        """
        return np.float64(self.Tau_ip(z, float(E)/(1000*GeV)))
        # return (z/3.3)*(E/(10*GeV))**.8 # Analytic approximation from 1506.05118

class LuminosityFunctionmAGN:
    """
    Class for calculating the luminosity function and source counts of mAGNs.
    Based on 1304.0908 (Di Mauro et al) and astro-ph/0010419 (Willot et al).
    """
    def __init__(self, data_dir = ''):
                
        self.set_params() # Set the radio LF parameters

        # Integration ranges
        self.Lmin = 10**41*erg*Sec**-1 # These are from the text after eq. (23)
        self.Lmax = 10**48*erg*Sec**-1
        self.zmin = 10**-3. # These are from the text after eq. (22) (using 10**-3 rather than 0)
        self.zmax = 4.
        self.Gammamin = 1.0
        self.Gammamax = 3.5

        self.Gamma_mean = 2.37 # Spectral index characteristics -- from text before equation (1) of Di Mauro et al
        self.Gamma_sigma = 0.32

        self.CTB_en_bins = 10**np.linspace(np.log10(0.3), np.log10(300),31) # CTB energy bins

        self.cd = cld.CosmologicalDistance() # For cosmological distance calculations -- default flat Lambda CDM parameters
        self.cdW = cld.CosmologicalDistance(omega_m = 0., omega_l = 0,h0=.5) # Cosmology used in Willot et al

        self.make_Tau()

    def set_params(self):
        """
        Parameters from the radio LF from Table 1 of Willot et al. Model C, omega_m = 0.
        """    
        self.rho_l0 = 10**-7.523*Mpc**-3
        self.alphal = 0.586
        self.Llstar = 10**26.48*W/Hz/sr
        self.zl0 = 0.710
        self.kl = 3.48
        self.rho_h0 = 10**-6.757*Mpc**-3
        self.alphah = 2.42
        self.Lhstar = 10**27.39*W/Hz/sr

        self.zh0 = 2.03
        self.zh1 = 0.568
        self.zh2 = 0.956

    def phi_R(self, LR151f, z):
        """
        Returns the Willot et al radio luminosity function (RLF), provided the radio luminosity at 151 MHz in 
        units of W/Hz/sr (LR151f) and redshift z. This is based on equations (7)-(14) of Willot et al.
        """

        if z < self.zl0:
            phi_l = self.rho_l0*(LR151f/self.Llstar)**-self.alphal*np.exp(-LR151f/self.Llstar)*(1+z)**self.kl
        if z >= self.zl0:
            phi_l = self.rho_l0*(LR151f/self.Llstar)**-self.alphal*np.exp(-LR151f/self.Llstar)*(1+self.zl0)**self.kl

        if z < self.zh0:
            phi_h = self.rho_h0*(LR151f/self.Lhstar)**-self.alphah*np.exp(-self.Lhstar/LR151f)*np.exp(-.5*((z-self.zh0)/self.zh1)**2)
        if z >= self.zh0:
            phi_h = self.rho_h0*(LR151f/self.Lhstar)**-self.alphah*np.exp(-self.Lhstar/LR151f)*np.exp(-.5*((z-self.zh0)/self.zh2)**2)

        eta = self.cdW.comoving_volume_element(z)/self.cd.comoving_volume_element(z) # ratio of differential volume slices to convert between cosmologies (see equation 14)

        return eta*(phi_l + phi_h)

    def phi_gamma(self, Lgamma, z):
        """
        Returns the gamma-ray LF given gamma-ray luminosity. Based on equation (20) of 1304.0908. First pre-factor 
        is from equation (5), second from equation (13). Post-factor converts to dN/dLdV form.

        Factor of 0.496 in the argument converts total radio luminosity from 5 GHz to 151 MHz. Divide by (151*MHz*sr) to get in appropriate 
        units [W/Hz/sr] for phi_R above.

        """
        return (1/1.008)*(1/0.77)*self.phi_R(((151*MHz)/(5*GHz))**-.8*self.LR5tot(self.LR5core(Lgamma))/(5*GHz*sr), z)*(1/(np.log(10)*Lgamma))

    def LR5tot(self, LR5core):
        """
        Returns total radio Luminosity at 5 GHz given core Luminosity at 5 GHz. Equation (13) of 
        Di Mauro et al and L ~ f**(-alpha+1) to convert 1.4 -> 5 GHz total luminosity (factor of 1.2899).
        """

        return 1.2899*(10**((np.log10(LR5core/(5*GHz)/(W/Hz))-4.2)/0.77))*(W/Hz)*(1.4*GHz)

    def LR5core(self, Lgamma):
        """
        Returns gamma-ray Luminosity given 5 GHz core radio Luminosity. Equation (8) of 1304.0908
        """

        return 10**((np.log10(Lgamma/(erg*Sec**-1))-2.00)/1.008)*(erg*Sec**-1)

    def Lgamma(self, Fgamma, Gamma,z):
        """
        Return luminosity flux given energy flux Fgamma
        """
        dL = self.cd.luminosity_distance(z)*Mpc

        E2 = 100*GeV
        E1 = 100*MeV
        
        N = Fgamma/((E2**(-Gamma+1)-E1**(-Gamma+1))/(-Gamma+1)) # Normalization for PL
        L_N = (4*np.pi*dL**2)*(E2**(-Gamma+2)-E1**(-Gamma+2))*(1+z)**(-2+Gamma)/(-Gamma+2) # Check it double deck it

        return N*L_N

    def dNdF(self, Fgamma):
        """
        Returns the differential source counts function
        """    
        dFgamma = Fgamma/1000

        return (1/dFgamma)*integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma: 4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z)*(1/(np.sqrt(2*np.pi)*self.Gamma_sigma))*np.exp(-(Gamma - self.Gamma_mean)**2/(2*self.Gamma_sigma**2)), self.Lgamma(Fgamma,Gamma,z), self.Lgamma(Fgamma+dFgamma,Gamma,z))[0], self.zmin,self.zmax)[0], self.Gammamin, self.Gammamax)[0]

    def NF(self, Fgamma):
        """
        Returns the cumulative source counts function
        """    

        return integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma: 4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z)*np.exp(-(Gamma - self.Gamma_mean)**2/(2*self.Gamma_sigma**2))*(1/(np.sqrt(2*np.pi)*self.Gamma_sigma)), self.Lgamma(Fgamma,Gamma,z), self.Lmax)[0], self.zmin,self.zmax)[0], self.Gammamin, self.Gammamax)[0]

    def dVdz(self,z):
        """
        Return comoving volument element
        """
        return self.cd.comoving_volume_element(z)*Mpc**3

    def make_Tau(self):
        """
        Create EBL interpolation (model based on Finke et al)
        """
        # Load and combine EBL files downloaded from http://www.phy.ohiou.edu/~finke/EBL/ 
        tau_files = self.data_dir + '/tau_modelC_total/tau_modelC_total_z%.2f.dat'
        z_list = np.arange(0, 5, 0.01) # z values to load files for
        E_list, tau_list = [],[]
        for z in z_list:
            d = np.genfromtxt(tau_files % z, unpack=True)
            E_list = d[0]
            tau_list.append(d[1])
        self.Tau_ip = ip.RectBivariateSpline(z_list, np.log10(E_list), np.log10(np.array(tau_list))) # Create interpolation

    def Tau(self,E,z):
        """
        EBL attenuation of gamma rays (mention references)
        """
        return np.float64(self.Tau_ip(E/(1000*GeV),z))
        # return (z/3.3)*(E/(10*GeV))**.8

    def dFdE(self, E,z, L, Gamma):
        """
        Intrinsic flux of source. Use simple power law.
        """
        E1 = 100*MeV
        E2 = 100*GeV
        dL = self.cd.luminosity_distance(z)*Mpc

        N = (1+z)**(2-Gamma)*L/(4*np.pi*dL**2)*(2-Gamma)/((1/E1)**-Gamma*(E2**(-Gamma+2)-E1**(-Gamma+2))) # Check it double deck it
        return N*((E/E1)**-Gamma)

    def opts0(self,*args, **kwargs):
        """
        Integration parameters
        """
        return {'epsrel':1e-2,'epsabs':0}
    
    def dIdE(self, E):
        """
        Return intensity spectrum of blazars. Since this is only used for sub-bin apportioning of photons, 
        we use a single index approximation (the source count function uses the full form)
        """

        Gamma = 2.37 # Assumed spectral index for mAGN

        self.dIdEval = integrate.nquad(lambda L,z: self.dVdz(z)*self.phi_gamma(L,z)*self.dFdE(E,z,L, Gamma)*(1/(np.sqrt(2*np.pi)*self.Gamma_sigma))*np.exp(-(Gamma - self.Gamma_mean)**2/(2*self.Gamma_sigma**2))*np.exp(-self.Tau(E,z)),[[self.Lmin,self.Lmax], [self.zmin, self.zmax]], opts=[self.opts0,self.opts0,self.opts0])[0]

        return self.dIdEval

    """
    ************************************************************
    Monte carlo integration experimentation
    """

    def dIdE_integrand(self, x):
        Gamma = x[0]
        L = x[1]
        z = x[2]
        return self.dVdz(z)*self.phi_gamma(L,z)*self.dFdE(self.E,z,L, Gamma)*(1/(np.sqrt(2*np.pi)*self.Gamma_sigma))*np.exp(-(Gamma - self.Gamma_mean)**2/(2*self.Gamma_sigma**2))*np.exp(-self.Tau(self.E,z))

    def dIdE_mc_vegas(self, E,nitn=10,neval=1e4):
        self.E = E

        integ = vegas.Integrator([[self.Gammamin,self.Gammamax], [self.Lmin,self.Lmax], [self.zmin,self.zmax]])
        result = integ(self.dIdE_integrand, nitn=nitn, neval=neval)
        print(result.summary())
        return result.mean

    """
    ************************************************************
    """

    def Fpgamma(self,Fgamma,E1,E2, Gamma):
        """
        Stretch flux for a given range to observed value over 0.1-100 GeV
        """
        
        self.E10 = 100*MeV
        self.E20 = 100*GeV
        return Fgamma*((self.E20**(-Gamma+1)-self.E10**(-Gamma+1))/(E2**(-Gamma+1)-E1**(-Gamma+1)))

    def dNdFp(self, Fgamma, E1, E2):
        """
        Returns the differential source counts function
        """    
        dFgamma = Fgamma/1000

        return (1/dFgamma)*integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma: 4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z)*(1/(np.sqrt(2*np.pi)*self.Gamma_sigma))*np.exp(-(Gamma - self.Gamma_mean)**2/(2*self.Gamma_sigma**2)), self.Lgamma(self.Fpgamma(Fgamma,E1,E2, Gamma),Gamma,z), self.Lgamma(self.Fpgamma(Fgamma+dFgamma,E1,E2, Gamma),Gamma,z))[0], self.zmin,self.zmax)[0], self.Gammamin, self.Gammamax)[0]

    def set_dIdE(self, Evals, dIdEvals):
        """
        Make interpolating function from calculated energy spectrum
        """
        self.dIdE_interp = ip.InterpolatedUnivariateSpline(Evals, dIdEvals)

class memoized(object):
   """
   Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated). 

   From https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize.
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)
