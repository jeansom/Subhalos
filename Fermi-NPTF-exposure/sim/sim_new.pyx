cimport numpy as np
#cimport healpy as hp
cimport cython

import sys, os

import numpy as np
from numpy.random import random_sample
import scipy
import matplotlib
import matplotlib.pyplot as plt

import healpy as hp
import mpmath as mp
from astropy.io import fits

from contextlib import contextmanager
import sys, os

#from tqdm import *

from scipy import integrate, interpolate

import pulsars.masks as masks
from config.make_NFW import make_NFW
from config.config_file import config 
import config.smooth_with_psf as swp
import config.compute_PSF as CPSF
#from config.analyze_results import analyze_results as ar

from fermi.fermi_plugin import fermi_plugin as fp

#from numba import jit, void, int_, double, autojit


cdef extern from "math.h":
    double log(double x) nogil
    double exp(double x) nogil
    double pow(double x, double y) nogil
    double sqrt(double x) nogil
    double abs(double x) nogil
    double fmin(double x, double y) nogil


def gauss(double r, double sigma):
    return gauss_int(r, sigma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double gauss_int(double r, double sigma):
    return r*exp(-r**2/(2*sigma**2))

class draw_points:
    def __init__(self,sigma,points,theta_center,phi_center,the_map):
        self.points = points
        self.sigma_rad=sigma*2*np.pi/360
        self.rand = random(self.r_dist)
        self.rand.draw_x_vec(points,0, np.max([10*self.sigma_rad,np.pi]),10**4)
        
        self.theta_center=theta_center
        self.phi_center = phi_center
        self.the_map = the_map
        self.nside = hp.npix2nside(len(self.the_map))
        
        self.find_rand_angle()
        self.put_counts_on_map()        
        
    def r_dist(self,r):
        return gauss(r, self.sigma_rad)
    
    def find_rand_angle(self):
        self.angle_vec = np.random.uniform(0,2*np.pi,self.points)
        dtheta = np.cos(self.angle_vec)*self.rand.x_vec
        self.theta_vec = dtheta+self.theta_center
        self.phi_vec = np.sin(self.angle_vec)*self.rand.x_vec/(np.sin(self.theta_center+dtheta/2))+self.phi_center
        
    def put_counts_on_map(self):
        theta_l_zero = np.where(self.theta_vec < 0)[0]
        if len(theta_l_zero) > 0:
            self.theta_vec[theta_l_zero] = - self.theta_vec[theta_l_zero]
            self.phi_vec[theta_l_zero] = - self.phi_vec[theta_l_zero]
        theta_g_zero = np.where(self.theta_vec > np.pi)[0]
        if len(theta_g_zero) > 0:
            self.theta_vec[theta_g_zero] = np.pi- (self.theta_vec[theta_g_zero] - np.pi)
            self.phi_vec[theta_g_zero] = - self.phi_vec[theta_g_zero]
        self.pix_vec = hp.ang2pix(self.nside,self.theta_vec,self.phi_vec)
        for pix in self.pix_vec:
            self.the_map[pix] += 1

def king_1quart(double r, double fcore, double score, double gcore, double stail, double gtail, double SpE):
    return full_king(r, fcore, score, gcore, stail, gtail, SpE)

def king_2quart(double r, double fcore1, double score1, double gcore1, double stail1, double gtail1, double SpE1, double fcore2, double score2, double gcore2, double stail2, double gtail2, double SpE2):
    return (1./2.)*full_king(r, fcore1, score1, gcore1, stail1, gtail1, SpE1)+(1./2.)*full_king(r, fcore2, score2, gcore2, stail2, gtail2, SpE2)

def king_3quart(double r, double fcore1, double score1, double gcore1, double stail1, double gtail1, double SpE1, double fcore2, double score2, double gcore2, double stail2, double gtail2, double SpE2, double fcore3, double score3, double gcore3, double stail3, double gtail3, double SpE3):
    return (1./3.)*full_king(r, fcore1, score1, gcore1, stail1, gtail1, SpE1)+(1./3.)*full_king(r, fcore2, score2, gcore2, stail2, gtail2, SpE2)+(1./3.)*full_king(r, fcore3, score3, gcore3, stail3, gtail3, SpE3)

def king_4quart(double r, double fcore1, double score1, double gcore1, double stail1, double gtail1, double SpE1, double fcore2, double score2, double gcore2, double stail2, double gtail2, double SpE2, double fcore3, double score3, double gcore3, double stail3, double gtail3, double SpE3, double fcore4, double score4, double gcore4, double stail4, double gtail4, double SpE4):
    return (1./4.)*full_king(r, fcore1, score1, gcore1, stail1, gtail1, SpE1)+(1./4.)*full_king(r, fcore2, score2, gcore2, stail2, gtail2, SpE2)++(1./4.)*full_king(r, fcore3, score3, gcore3, stail3, gtail3, SpE3)+(1./4.)*full_king(r, fcore4, score4, gcore4, stail4, gtail4, SpE4)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double basic_king(double x, double sigma, double gamma):
    return (1/(2*np.pi*sigma**2))*(1-1/gamma)*(1+x**2/(2*gamma*sigma**2))**(-gamma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double full_king(double r, double fcore, double score, double gcore, double stail, double gtail, double SpE):
    return 2*np.pi*r*(fcore*basic_king(r/SpE,score,gcore)+(1-fcore)*basic_king(r/SpE,stail,gtail))

class draw_points_king:
    """ Analogous to draw_points, but for a king function """
    def __init__(self,fcore,score,gcore,stail,gtail,SpE,points,theta_center,phi_center,the_map):
        self.points = points
        self.fcore = fcore
        self.score = score
        self.gcore = gcore
        self.stail = stail
        self.gtail = gtail
        self.SpE = SpE
        if (len(fcore) == 1):
            self.rand = random(self.r_dist_1quart)
            self.rand.draw_x_vec(points,0, np.max([10*SpE[0]*(score[0]+stail[0])/2.,np.pi]),10**4)
        if (len(fcore) == 2):
            self.rand = random(self.r_dist_2quart)
            self.rand.draw_x_vec(points,0, np.max([10*SpE[1]*(score[1]+stail[1])/2.,np.pi]),10**4)
        if (len(fcore) == 3):
            self.rand = random(self.r_dist_3quart)
            self.rand.draw_x_vec(points,0, np.max([10*SpE[2]*(score[2]+stail[2])/2.,np.pi]),10**4)
        if (len(fcore) == 4):
            self.rand = random(self.r_dist_4quart)
            self.rand.draw_x_vec(points,0, np.max([10*SpE[3]*(score[3]+stail[3])/2.,np.pi]),10**4)

        self.theta_center=theta_center
        self.phi_center = phi_center
        self.the_map = the_map
        self.nside = hp.npix2nside(len(self.the_map))

        self.find_rand_angle()
        self.put_counts_on_map()

    def r_dist_1quart(self,r):
        return king_1quart(r, self.fcore[0], self.score[0], self.gcore[0], self.stail[0], self.gtail[0], self.SpE[0])

    def r_dist_2quart(self,r):
        return king_2quart(r, self.fcore[0], self.score[0], self.gcore[0], self.stail[0], self.gtail[0], self.SpE[0], self.fcore[1], self.score[1], self.gcore[1], self.stail[1], self.gtail[1], self.SpE[1])

    def r_dist_3quart(self,r):
        return king_3quart(r, self.fcore[0], self.score[0], self.gcore[0], self.stail[0], self.gtail[0], self.SpE[0], self.fcore[1], self.score[1], self.gcore[1], self.stail[1], self.gtail[1], self.SpE[1], self.fcore[2], self.score[2], self.gcore[2], self.stail[2], self.gtail[2], self.SpE[2])

    def r_dist_4quart(self,r):
        return king_4quart(r, self.fcore[0], self.score[0], self.gcore[0], self.stail[0], self.gtail[0], self.SpE[0], self.fcore[1], self.score[1], self.gcore[1], self.stail[1], self.gtail[1], self.SpE[1], self.fcore[2], self.score[2], self.gcore[2], self.stail[2], self.gtail[2], self.SpE[2], self.fcore[3], self.score[3], self.gcore[3], self.stail[3], self.gtail[3], self.SpE[3])

    def find_rand_angle(self):
        self.angle_vec = np.random.uniform(0,2*np.pi,self.points)
        dtheta = np.cos(self.angle_vec)*self.rand.x_vec
        self.theta_vec = dtheta+self.theta_center
        self.phi_vec = np.sin(self.angle_vec)*self.rand.x_vec/(np.sin(self.theta_center+dtheta/2))+self.phi_center

    def put_counts_on_map(self):
        theta_l_zero = np.where(self.theta_vec < 0)[0]
        if len(theta_l_zero) > 0:
            self.theta_vec[theta_l_zero] = - self.theta_vec[theta_l_zero]
            self.phi_vec[theta_l_zero] = - self.phi_vec[theta_l_zero]
        theta_g_zero = np.where(self.theta_vec > np.pi)[0]
        if len(theta_g_zero) > 0:
            self.theta_vec[theta_g_zero] = np.pi- (self.theta_vec[theta_g_zero] - np.pi)
            self.phi_vec[theta_g_zero] = - self.phi_vec[theta_g_zero]
        self.pix_vec = hp.ang2pix(self.nside,self.theta_vec,self.phi_vec)
        for pix in self.pix_vec:
            self.the_map[pix] += 1

class fake_data:
    def __init__(self):

        self.ps_spectra = []
        self.ps_maps = []
        self.exact_ps_maps = []
        self.low_flux_maps_total = []

    def init_fake_data(self,nside,En_bins,template_dict,mask_total='Null'):
        self.nside=nside
        self.CTB_en_bins = En_bins
        self.CTB_en_min = self.CTB_en_bins[0]
        self.CTB_en_max = self.CTB_en_bins[-1]
        self.nEbins = len(self.CTB_en_bins)

        self.ps_spectra = []
        self.ps_maps = []
        self.exact_ps_maps = []

        self.template_dict = template_dict
        self.mask_total = mask_total

    def make_plane_mask(self, band_mask_range):
        plane_mask_array = masks.mask_lat_band(90+band_mask_range[0], 90+band_mask_range[1], self.nside)
        return plane_mask_array

    def make_long_mask(self, lmin, lmax):
        long_mask_array = masks.mask_not_long_band(lmin, lmax, self.nside)
        return long_mask_array

    def make_lat_mask(self, bmin, bmax):
        lat_mask_array = masks.mask_not_lat_band(90+bmin, 90+bmax, self.nside)
        return lat_mask_array

    def make_ring_mask(self, inner, outer, lat, lng):
        ring_mask = np.logical_not(masks.mask_ring(inner, outer, lat, lng, self.nside))
        return ring_mask

    def make_mask_total(self, plane_mask = True, band_mask_range = [-30,30], lcut=False, lmin=-20, lmax=20,bcut=False, bmin=-20, bmax=20, mask_ring=False, inner=0, outer=30,lat= 90, lng = 0, ps_mask_array = 'False'):
        """ Combine various masks into a single boolean array where True pixels are masked """
        # First add all variables to self as use some of them later
        self.plane_mask = plane_mask
        self.band_mask_range = band_mask_range
        self.lcut = lcut
        self.lmin = lmin
        self.lmax = lmax
        self.bcut = bcut
        self.bmin = bmin
        self.bmax = bmax
        self.mask_ring = mask_ring
        self.inner = inner
        self.outer = outer
        self.lat = lat
        self.lng = lng
        self.ps_mask_array = ps_mask_array

        # Now compute all the masks we might need
        self.plane_mask_ar = self.make_plane_mask(self.band_mask_range)
        self.long_mask_ar = self.make_long_mask(self.lmin, self.lmax)
        self.lat_mask_ar = self.make_lat_mask(self.bmin, self.bmax)
        self.ring_mask = self.make_ring_mask(self.inner, self.outer, self.lat, self.lng)

        # Next initialise a healpix array of false values as only true values are masked
        mask_array = np.zeros(self.nside**2*12, dtype=bool)

        if self.plane_mask:
            mask_array += self.plane_mask_ar

        if self.lcut:
            mask_array += self.long_mask_ar

        if self.bcut:
            mask_array += self.lat_mask_ar

        if self.mask_ring:
            mask_array += self.ring_mask

        self.mask_geom_total = mask_array

        # Finally add in a point source mask if this was included
        if self.ps_mask_array == 'False':
            self.mask_total = self.mask_geom_total
            self.mask_total_array = np.array([self.mask_geom_total for en in self.CTB_en_bins])
        else:
            self.mask_total = np.sum(self.ps_mask_array,axis=0,dtype=bool)+self.mask_geom_total
            self.mask_total_array = np.array(self.ps_mask_array + self.mask_geom_total,dtype=bool)
    
    def init_back(self,A):
        self.use_ps=False
        self.A_vec = A
        self.xbg = [ np.sum(np.array([self.template_dict[A[i]][en] for i in range(len(self.A_vec))]),axis=0) for en in range(len(self.CTB_en_bins)-1) ]

    def init_ps_from_function(self,ps_profile,S_array,dN_array,ps_spectrum,psf_array,n_ps = 10000,simplify_ps_map=False,nside_detail_factor=8*2,smin=0.001,smax=2000.,n_bins=100000.0, exposure_map= None):
        print 'The number of energy bins is ', len(ps_spectrum) 
        self.ps_spectra.append( ps_spectrum )
        self.simplify_ps_map=simplify_ps_map
        self.use_ps = True
        if exposure_map is None:
            self.ps_profile = ps_profile
            self.exposure_map=exposure_map
        else:
            # Increasing the npix of the exposure maps
            self.exposure_map = hp.ud_grade(exposure_map,self.nside*nside_detail_factor,power=-2)
            # Averaging out the exposure maps
            self.exposure_map_mean = np.mean(self.exposure_map, axis=0)  # The actual exposure map f.CTB_exposure_maps,
            self.exposure_mean = np.mean(self.exposure_map)
            # Dividing the ps_profile by the exposure
            # First making a non-amplified version of exposure_map_mean
            self.exposure_map_mean_original_nside = np.mean(exposure_map, axis=0) 
            self.exposure_mean_original_nside = np.mean(exposure_map)
            flux_profile = ps_profile / self.exposure_map_mean_original_nside
            self.ps_profile = flux_profile*self.exposure_mean_original_nside
            self.ps_profile /= np.mean(self.ps_profile)

        self.S_array = np.array(S_array)
        self.dN_array = np.array(dN_array)


        self.find_counts_from_function(smin)
        self.gpd = generate_ps_data_from_function(self.ps_profile,self.S_array,self.dN_array,nside_detail_factor=nside_detail_factor, exposure_map = self.exposure_map)
        self.gpd.config_source_count(self.counts,n_ps=n_ps,smin = smin)
        if exposure_map is None:
            self.gpd.assign_counts()
        else:
            self.gpd.assign_counts_exposure()
        if self.simplify_ps_map:
            self.gpd.simplify_counts_map(self.mask_total)

        temp_map_array = []
        #temp_exact_map_array = []
        for psf,spect in map(None,psf_array,ps_spectrum):
            print 'The psf is ', psf
            print 'The spect coefficeint is ', spect
            self.gpd.smear_map(psf,spectra_factor=spect)
            temp_map_array.append(self.gpd.counts_map_psf)
            #temp_exact_map_array.append(self.gpd.counts_map_detail*spect) #was self.gps.counts_map
        self.gpd.cleanup()
        self.ps_maps.append(temp_map_array)
        self.exact_ps_maps.append(self.gpd.counts_map_detail)
        self.make_low_flux_ps_data(ps_spectrum)

    def init_ps(self,ps_profile,theta_PS,psf_array,Sc=10000.,n_ps = 10000,simplify_ps_map=True,nside_detail_factor=8*2,smin=0.001,smax=1000.,n_bins=100000.0, exposure_map= None):
        theta_PS = list(theta_PS)
        Sb_total=float(np.sum(theta_PS[3:]))
        print 'The number of energy bins is ', len(theta_PS[3:])
        A_total = theta_PS[0]/Sb_total # Goes back from A tilde to A.
        # if len(theta_PS[3:])>1:
        #     A_total = theta_PS[0]/Sb_total
        # else:
        #     A_total = theta_PS[0]
        theta_PS_total = [A_total, theta_PS[1],theta_PS[2], Sb_total]
        ps_spectrum=np.array(theta_PS[3:]) / Sb_total 
        self.ps_spectra.append( ps_spectrum )
        self.simplify_ps_map=simplify_ps_map
        self.use_ps = True

        if exposure_map is None:
            self.ps_profile = ps_profile
            self.exposure_map=exposure_map
        else:
            # Increasing the npix of the exposure maps
            self.exposure_map = hp.ud_grade(exposure_map,self.nside*nside_detail_factor,power=-2)
            # Averaging out the exposure maps
            self.exposure_map_mean = np.mean(self.exposure_map, axis=0)  # The actual exposure map f.CTB_exposure_maps,
            self.exposure_mean = np.mean(self.exposure_map)
            # Dividing the ps_profile by the exposure
            # First making a non-amplified version of exposure_map_mean
            self.exposure_map_mean_original_nside = np.mean(exposure_map, axis=0) 
            self.exposure_mean_original_nside = np.mean(exposure_map)
            flux_profile = ps_profile / self.exposure_map_mean_original_nside
            self.ps_profile = flux_profile*self.exposure_mean_original_nside 

        self.A, self.n1, self.n2 , self.Sb = theta_PS_total
        print 'A, n1, n2, Sb: ', self.A, self.n1, self.n2 , self.Sb
        self.Sc = Sc
        self.find_counts(smin)
        self.gpd = generate_ps_data(self.ps_profile,self.A,self.n1,self.n2,self.Sb,Sc=self.Sc,nside_detail_factor=nside_detail_factor, exposure_map = self.exposure_map)
        self.gpd.config_source_count(self.counts,n_ps=n_ps,smin = smin,smax = smax, n_bins = n_bins)
        if exposure_map is None:
            self.gpd.assign_counts()
        else:
            self.gpd.assign_counts_exposure()
        if self.simplify_ps_map:
            self.gpd.simplify_counts_map(self.mask_total)

        temp_map_array = []
        #temp_exact_map_array = []
        for psf,spect in map(None,psf_array,ps_spectrum):
            print 'The psf is ', psf
            print 'The spect coefficeint is ', spect
            self.gpd.smear_map(psf,spectra_factor=spect)
            temp_map_array.append(self.gpd.counts_map_psf)
            #temp_exact_map_array.append(self.gpd.counts_map_detail*spect) #was self.gps.counts_map
        self.gpd.cleanup()
        self.ps_maps.append(temp_map_array)
        self.exact_ps_maps.append(self.gpd.counts_map_detail)
        self.make_low_flux_ps_data(ps_spectrum)

    def init_ps_king(self,ps_profile,theta_PS,fcore_array,score_array,gcore_array,stail_array,gtail_array,SpE_array,Sc=10000.,n_ps = 10000,simplify_ps_map=True,nside_detail_factor=8*2,smin=0.001,smax=1000.,n_bins=100000.0):
        theta_PS = list(theta_PS)
        Sb_total=float(np.sum(theta_PS[3:]))
        print 'The number of energy bins is ', len(theta_PS[3:])
        A_total = theta_PS[0]/Sb_total
        # if len(theta_PS[3:])>1:
        #     A_total = theta_PS[0]/Sb_total
        # else:
        #     A_total = theta_PS[0]
        theta_PS_total = [A_total, theta_PS[1],theta_PS[2], Sb_total]
        ps_spectrum=np.array(theta_PS[3:]) / Sb_total
        self.ps_spectra.append( ps_spectrum )
        self.simplify_ps_map=simplify_ps_map
        self.use_ps = True
        self.ps_profile = ps_profile
        self.A, self.n1, self.n2 , self.Sb = theta_PS_total
        print 'A, n1, n2, Sb: ', self.A, self.n1, self.n2 , self.Sb
        self.Sc = Sc
        self.find_counts(smin)
        self.gpd = generate_ps_data(self.ps_profile,self.A,self.n1,self.n2,self.Sb,Sc=self.Sc,nside_detail_factor=nside_detail_factor)
        self.gpd.config_source_count(self.counts,n_ps=n_ps,smin = smin,smax = smax, n_bins = n_bins)
        self.gpd.assign_counts()
        if self.simplify_ps_map:
            self.gpd.simplify_counts_map(self.mask_total)

        temp_map_array = []
        #temp_exact_map_array = []
        for fcore,score,gcore,stail,gtail,SpE,spect in map(None,fcore_array,score_array,gcore_array,stail_array,gtail_array,SpE_array,ps_spectrum):
            self.gpd.smear_map_king(fcore,score,gcore,stail,gtail,SpE,spectra_factor=spect)
            temp_map_array.append(self.gpd.counts_map_psf)
            #temp_exact_map_array.append(self.gpd.counts_map_detail*spect) #was self.gps.counts_map
        self.gpd.cleanup()
        self.ps_maps.append(temp_map_array)
        self.exact_ps_maps.append(self.gpd.counts_map_detail)
        self.make_low_flux_ps_data(ps_spectrum)

    def find_counts(self,smin):
        # Siddharth: added else for Sb < smin, may be finicky -- make sure things make sense if this is the case
        if self.Sb > smin:
            self.lower_counts =np.sum(self.ps_profile)*self.A*(1 / (2-self.n2)) * (smin**2)*(smin/self.Sb)**(-self.n2)
            self.counts = np.sum(self.ps_profile)*self.A*(self.Sb**2)*( 1/(2.-self.n2) - (1 - (self.Sb**(self.n1-2))*(self.Sc**(2-self.n1)))/(2.-self.n1)) - self.lower_counts
        else:
            self.lower_counts = int(np.sum(self.ps_profile)*integrate.quad(lambda S: S*self.broken_power_law(S, self.A, self.n1, self.n2, self.Sb), 0.0001,smin)[0])
            self.counts = int(np.sum(self.ps_profile)*integrate.quad(lambda S: S*self.broken_power_law(S, self.A, self.n1, self.n2, self.Sb), smin,self.Sc)[0])

        print 'The number of counts (from simulated PSs) should be ', self.counts
        print 'The number of counts (from non-simulated PSs) should be ', self.lower_counts
        print 'The number of counts (from total PSs) should be ', self.counts + self.lower_counts

    def find_counts_from_function(self,smin):
        self.S_array_min_index = np.where(self.S_array>smin)[0][0]
        self.lower_counts = np.sum(self.ps_profile)*np.sum( self.dN_array[:self.S_array_min_index]*self.S_array[:self.S_array_min_index]   )
        self.counts = np.sum(self.ps_profile)*np.sum( self.dN_array[self.S_array_min_index:]*self.S_array[self.S_array_min_index:]   )

        print 'The number of counts (from simulated PSs) should be ', self.counts
        print 'The number of counts (from non-simulated PSs) should be ', self.lower_counts
        print 'The number of counts (from total PSs) should be ', self.counts + self.lower_counts

    def make_low_flux_ps_data(self,ps_spectrum):
        
        if self.exposure_map is None:
            self.low_flux_smooth_maps = np.array([self.ps_profile * self.lower_counts / float(np.sum(self.ps_profile)) * spect for spect in ps_spectrum]) # Siddharth: fixed type multiplication and division bug
        else:
            self.low_flux_smooth_maps = np.array([self.ps_profile * self.lower_counts / float(np.sum(self.ps_profile)) * spect * self.exposure_map_mean_original_nside / self.exposure_mean_original_nside for spect in ps_spectrum])
        
        self.low_flux_data_maps = np.array([ np.random.poisson(the_map) for the_map in self.low_flux_smooth_maps ])
        self.low_flux_maps_total.append(self.low_flux_data_maps)

    # def init_model(self):
    #     if self.use_ps:
    #         self.total_ps = np.sum(self.ps_maps,axis=0) 
    #         self.model = self.xbg + self.total_ps
    #     else:   
    #         self.model = self.xbg

    def make_fake_data(self):
        #self.init_model()
        for en in range(len(self.xbg)):
            xbg = self.xbg[en]
            if len(np.where(xbg < 0))>0:
                print 'Warning: mean below zero in a pixel at energy bin index ', en
                self.xbg[en][np.where(xbg<0)]=np.zeros(len(np.where(xbg<0)))
                print 'fixed the problem by setting mean to zero counts, but be careful!'
        self.fake_data = [np.random.poisson(xbg) for xbg in self.xbg]
        if self.use_ps:
            print 'there are', len(self.ps_maps), 'populations of sources being added and saved'
            self.total_ps = np.sum(self.ps_maps,axis=0)
            self.total_ps_low_flux = np.sum(self.low_flux_maps_total,axis=0)

            print 'low and high flux total counts are', np.sum(self.low_flux_maps_total, axis=1), np.sum(self.ps_maps, axis=1)
            
            for en in range(len(self.xbg)):
                self.fake_data[en] += self.total_ps[en] + self.total_ps_low_flux[en]

    def save_fake_data(self,fake_data_path = 'data/fake_data/test.txt.gz'):
        # if not os.path.exists(fake_data_dir):
        #     os.mkdir(fake_data_dir)
        np.savetxt(fake_data_path, self.fake_data)

        # if self.use_ps:
        #     np.savetxt(fake_data_dir + fake_data_tag, self.fake_data)
        # else:
        #     np.savetxt(fake_data_dir + fake_data_tag, [self.fake_data,self.model])

    def save_fake_data_ps_only(self,fake_data_ps_path = 'data/fake_data/test.txt.gz'):
        # if not os.path.exists(fake_data_dir):
        #     os.mkdir(fake_data_dir)
        np.savetxt(fake_data_ps_path, self.total_ps)

        # if self.use_ps:
        #     np.savetxt(fake_data_dir + fake_data_tag, self.fake_data)
        # else:
        #     np.savetxt(fake_data_dir + fake_data_tag, [self.fake_data,self.model])

    def save_fake_data_low_flux_only(self,fake_data_low_flux_path = 'data/fake_data/test_low_flux.txt.gz'):
        # if not os.path.exists(fake_data_dir):
        #     os.mkdir(fake_data_dir)
        np.savetxt(fake_data_low_flux_path, self.total_ps_low_flux)

        # if self.use_ps:
        #     np.savetxt(fake_data_dir + fake_data_tag, self.fake_data)
        # else:
        #     np.savetxt(fake_data_dir + fake_data_tag, [self.fake_data,self.model])

    def save_fake_data_key(self,fake_data_key_path='data/fake_data/test_key.txt.gz'):
        '''Fake data key will already be exposure corrected back to flux style'''
        counts_array=[]
        for counts_map, spectra, k in map(None,self.exact_ps_maps,self.ps_spectra, range(len(self.exact_ps_maps)) ):
            if self.exposure_map is None:
                rescale = np.ones( len(counts_map)  )
            else:
                rescale = self.exposure_mean / self.exposure_map_mean
            print 'shapes are', np.shape(self.exposure_map_mean), np.shape(counts_map)

            where_vec=np.where(counts_map>0)[0]
            counts_map_rescale = counts_map * rescale
            counts_non_zero=counts_map_rescale[where_vec]
            nside_temp = hp.npix2nside(len(counts_map))
            theta_phi_array=np.array(hp.pix2ang(nside_temp,where_vec))
            counts_array_temp= [[theta_phi_array[0,i],theta_phi_array[1,i]] + list(counts_non_zero[i]*spectra) for i in range(len(where_vec)) ]
            counts_array+=list(counts_array_temp)
            print 'doing population number', i, 'with', np.sum(counts_non_zero), 'photons'
            # print 'shape temp: ', np.shape(counts_array_temp) 
            # print 'shape counts_array: ', np.shape(counts_array) 
        #counts_array = np.array(np.array([ca_0 for ca_1 in counts_array for ca_0 in ca_1]))
        self.fake_data_key_path = fake_data_key_path
        self.exact_ps_maps_total = counts_array #np.sum( np.array(self.exact_ps_maps),axis=0 )
        # print self.exact_ps_maps_total
        # print np.shape(self.exact_ps_maps_total)
        np.savetxt(fake_data_key_path,self.exact_ps_maps_total)


    # def histogram_fake_ps_data(self,exposure_maps,*args,**kwargs):
    #     import analysis_classes.sim_analysis as sa
    #     fake3FGL = sa.make_flux_histogram(self.fake_data_key_path,exposure_maps,band_mask_range = [-mask_b_plot,mask_b_plot], mask_ring = not(high_lat), outer = outer_plot)
    #     pfake3FGL.make_fake_data_flux_histogram(0.1,500,25,emin,emax)
    #     print 'emin, emax: ', emin, emax
    #     pfake3FGL.plot_fake_data_histogram(fmt = 'o', color='blue',markersize=5,label='sim PS')

class fake_fermi_data(fp,fake_data):
    def __init__(self,*args,**kwargs):
        fp.__init__(self,*args,**kwargs)
        fake_data.__init__(self)

    def rebin_external_data(self,nEbins):
        bin_len = (len(self.CTB_en_bins)-1)/nEbins
        self.CTB_en_bins = np.array( [self.CTB_en_bins[bin_len*i] for i in range(nEbins+1)]   )
        self.CTB_en_min = self.CTB_en_bins[0]
        self.CTB_en_max = self.CTB_en_bins[-1]
        self.nEbins = len(self.CTB_en_bins)

        self.CTB_exposure_maps = np.array([np.mean(self.CTB_exposure_maps[bin_len*i:bin_len*(i+1)],axis=0) for i in range(nEbins)])
        self.CTB_count_maps = np.array([np.sum(self.CTB_count_maps[bin_len*i:bin_len*(i+1)],axis=0) for i in range(nEbins)])

        self.total_exposure_map = np.mean(self.CTB_exposure_maps,axis=0)
        self.NPIX_total = np.size(self.total_exposure_map)

        for comp in self.template_dict.keys():
            self.template_dict[comp] = np.array([np.sum(self.template_dict[comp][bin_len*i:bin_len*(i+1)],axis=0) for i in range(nEbins)])

# class fake_data:
#     def __init__(self,nside,En_bins,template_dict,use_ps=False,psf_array='Null',theta_PS='Null',ps_profile='Null',mult_sigma_for_smooth=5,nside_smear_factor=4,simplify_ps_map=False,Sc=10000.,n_ps = 10000,mask_total='Null'):
#         self.nside=nside
#         self.CTB_en_bins = En_bins
#         self.CTB_en_min = self.CTB_en_bins[0]
#         self.CTB_en_max = self.CTB_en_bins[-1]
#         self.nEbins = len(self.CTB_en_bins)

#         self.ps_maps = []

#         self.template_dict = template_dict
#         self.use_ps=use_ps
#         self.init_back()
#         if self.use_ps:
#             self.psf_array=psf_array
#             self.theta_PS = theta_PS
#             self.ps_profile = ps_profile
#             self.mult_sigma_for_smooth, self.nside_smear_factor, self.simplify_ps_map = mult_sigma_for_smooth, nside_smear_factor,simplify_ps_map
#             self.Sc, self.n_ps = Sc, n_ps
#             self.mask_total = mask_total
#             self.init_ps()

#         self.init_model()
#         self.make_fake_data()

#     def init_back(self):
#         A = self.template_dict.keys()
#         self.xbg = [ np.sum(np.array([self.template_dict[A[i]][en] for i in range(len(A))]),axis=0) for en in range(len(self.CTB_en_bins)-1) ]

#     def init_ps(self):
#         theta_PS = list(self.theta_PS)
#         theta_PS_total = theta_PS[0:3] + [np.sum(theta_PS[3:])]
#         ps_spectrum=np.array(theta_PS[3:]) / float(np.sum(theta_PS[3:])) 
#         #self.ps_spectra.append( ps_spectrum )
#         self.A, self.n1, self.n2 , self.Sb = theta_PS_total
#         print self.A,self.n1,self.n2,self.Sb
#         self.find_counts()
#         self.gpd = generate_ps_data(self.ps_profile,self.A,self.n1,self.n2,self.Sb,self.Sc)
#         self.gpd.config_source_count(self.counts,n_ps=self.n_ps)
#         self.gpd.assign_counts()
#         if self.simplify_ps_map:
#             self.gpd.simplify_counts_map(self.mask_total)

#         temp_map_array = []
#         for psf,spect in map(None,self.psf_array,ps_spectrum):
#             print 'The psf is ', psf
#             self.gpd.smear_map(psf,self.mult_sigma_for_smooth,nside_smear_factor=self.nside_smear_factor)
#             temp_map_array.append(self.gpd.counts_map_psf*spect)
#         self.ps_maps.append(temp_map_array)

#     def find_counts(self):
#         self.counts = np.sum(self.ps_profile)*self.A*(self.Sb**2)*( 1/(2.-self.n2) - (1 - (self.Sb**(self.n1-2))*(self.Sc**(2-self.n1)))/(2.-self.n1))
#         print 'The number of counts should be ', self.counts

#     def init_model(self):
#         if self.use_ps:
#             self.total_ps = np.sum(self.ps_maps,axis=0) 
#             self.model = self.xbg + self.total_ps
#         else:   
#             self.model = self.xbg

#     def make_fake_data(self):
#         self.init_model()
#         self.fake_data = [np.random.poisson(mod) for mod in self.model]

#     def save_fake_data(self,fake_data_path = 'data/fake_data/test.txt.gz'):
#         # if not os.path.exists(fake_data_dir):
#         #     os.mkdir(fake_data_dir)
#         np.savetxt(fake_data_path, self.fake_data)

# class fake_data_from_file(fake_fermi_data):
#     def __init__(self,name ='ben',tag = 'test',run_tag = 'test',nside = 128):

#         self.name = name
#         self.dict_dir = 'dict/'+tag + '/'#+ run_tag + '/'
#         self.run_tag = run_tag
#         self.tag = tag

#         self.a = ar(self.dict_dir,self.run_tag)

#         self.CTB_en_min, self.CTB_en_max = self.a.the_dict['CTB_en_min'], self.a.the_dict['CTB_en_max']
#         #self.nside = self.a.the_dict['nside']
#         self.nside = nside
#         self.data_name = self.a.the_dict['data_name']

#         fake_data.__init__(self,name=self.name,tag = self.tag,CTB_en_min = self.CTB_en_min, CTB_en_max = self.CTB_en_max, nside = self.nside,data_name = self.data_name)

#         print 'The model parameters are ', self.a.the_dict['params']
#         #print 'Please upload templates.'


#     def config_back_from_file(self):
#         A_log = self.a.return_poiss_medians()
#         A = self.convert_log_list(A_log,self.a.the_dict['poiss_list_is_log_prior'])
#         self.summed_templates_raw = self.a.the_dict['summed_templates_not_compressed']

#         self.init_back(A,summed_templates = [hp.ud_grade(temp,self.nside,power=-2) for temp in self.summed_templates_raw])

#     def config_ps_from_file(self,temp_number,psf = 'False',mult_sigma_for_smooth = 5,Sc = 100000.,n_ps = 100000,simplify_ps_map = False):
#         self.simplify_ps_map = simplify_ps_map
#         self.Sc = Sc
#         a = self.a

#         self.ps_template = self.summed_templates[temp_number]
#         if psf == 'False':
#             self.psf = CPSF.main(a.the_dict['CTB_en_bins'], a.the_dict['spect'], self.nside, a.the_dict['psf_dir'], just_sigma = True)
#         else:
#             self.psf = psf

#         self.log_theta_ps = [a.the_dict['s']['marginals'][a.the_dict['n_poiss']+i]['median'] for i in range(0,a.the_dict['n_non_poiss'])]


#         self.theta_ps = self.convert_log_list(self.log_theta_ps,a.the_dict['non_poiss_list_is_log_prior_uncompressed'][0])

#         self.init_ps(self.ps_template,self.theta_ps,psf = self.psf,Sc=self.Sc,n_ps = n_ps,mult_sigma_for_smooth = mult_sigma_for_smooth)


class random:
    def __init__(self,func):
        self.func = func
        
    def make_x_array(self,xmin, xmax ,n_bins):
        self.x_array = np.linspace(xmin,xmax,n_bins)
        self.dx = self.x_array[1]-self.x_array[0]
        
    def make_prob_vec(self,xmin, xmax ,n_bins):
        self.make_x_array(xmin , xmax, n_bins)
        self.probabilities = np.vectorize(self.func)(self.x_array)
    
    def draw_x_vec(self,n_ps,xmin, xmax,n_bins):
        self.n_ps = n_ps
        self.make_prob_vec(xmin,xmax,n_bins)
        self.weighted_values()
        
    def weighted_values(self):
        probabilities = self.probabilities/float(np.sum(self.probabilities))
        bins = np.add.accumulate(probabilities)
        self.x_vec = self.x_array[np.searchsorted(bins,random_sample(self.n_ps))]


class source_count(random):
    def __init__(self,A,n1,n2,Sb,Sc):
        self.A, self.n1,self.n2,self.Sb,self.Sc = float(A),float(n1),float(n2),float(Sb),float(Sc)
        print 'In source_count: A, n1,n2, Sb, Sc are ', self.A, self.n1,self.n2,self.Sb,self.Sc
        random.__init__(self,self.dnds)
    
    def dnds(self,s):
        if s <= self.Sb:
            n_ps= self.A*(s/self.Sb)**-self.n2
            # return self.A*(s/self.Sb)**-self.n2
        elif s > self.Sb and s < self.Sc:
            n_ps= self.A*(s/self.Sb)**-self.n1
        else:
            n_ps =  0.0
        #print n_ps
        return n_ps
        
    def find_count_list(self,max_counts,n_ps = 10000, smin = 0.001,smax = 1000., n_bins = 100000.):
        self.draw_x_vec(n_ps,smin,smax,n_bins)
        self.i_stop = next(i for i in range(n_ps) if  np.sum(self.x_vec[0:i]) > max_counts)
        self.count_list = self.x_vec[0:self.i_stop]

        self.n_ps_real = len(self.count_list)
        print 'The number of PSs is going to be ', self.n_ps_real
        self.total_counts = np.sum(self.count_list)


class source_count_from_function(random):
    def __init__(self,S_array,dN_array):
        self.S_array, self.dN_array = S_array,dN_array
        random.__init__(self,None)
        self.x_array = S_array
        self.probabilities = dN_array
        
    def find_count_list(self,max_counts,n_ps = 10000, smin = 0.001):
        self.n_ps = n_ps
        where_vec = np.where(self.S_array > smin)[0]
        self.S_array = self.S_array[where_vec]
        self.dN_array = self.dN_array[where_vec]
        self.x_array = self.S_array
        self.probabilities = self.dN_array


        self.count_list = []
        for N, S in zip(self.dN_array,self.S_array):
            num = np.random.poisson(N)
            print N, num, S
            self.count_list += num*[S]

        self.count_list = np.array(self.count_list)
        np.random.shuffle(self.count_list)
        # self.weighted_values()

        # self.i_stop = next(i for i in range(n_ps) if  np.sum(self.x_vec[0:i]) > max_counts)
        # self.count_list = self.x_vec[0:self.i_stop]

        self.n_ps_real = len(self.count_list)
        print 'The number of PSs is going to be ', self.n_ps_real
        self.total_counts = np.sum(self.count_list)



class source_spatial_dist(random):
    def __init__(self,the_map):
        self.the_map = the_map
        self.npix = len(self.the_map)
        self.nside = hp.npix2nside(self.npix)
        
        random.__init__(self,self.spatial_func)
        
    def spatial_func(self,i):
        return self.the_map[int(i)]
    
    def find_spatial_list(self,n_ps = 10000):
        print 'making spatial list, and the number of pixels is ', self.npix
        self.draw_x_vec(n_ps,0,self.npix-1,self.npix)
        print 'finished drawing x vec!'
        self.pos_vec = np.vectorize(int)(self.x_vec)
        self.the_ps_map = np.histogram(self.pos_vec,bins=np.arange(0,self.npix+1))[0]
        print 'finished making spatial list.'
        
    def plot_spatial_dist(self,lonra = [-10,10],latra = [-10,10]):
        plot = hp.cartview(self.the_ps_map,lonra = lonra,latra = latra)
        return plot   



class generate_ps_data():
    
    def __init__(self,the_map,A,n1,n2,Sb,Sc=10000.,nside_detail_factor=16, exposure_map=None): #have the keyword here for using exposure.  exposure = None here
        self.nside_detail_factor=nside_detail_factor
        self.nside = hp.npix2nside(len(the_map))
        self.the_map_detail = hp.ud_grade(the_map,nside_detail_factor*self.nside,power=-2)
        self.ssd = source_spatial_dist(self.the_map_detail) #here you want to divide self.the_map_detail by the exposure map * mean of exposure map.  #the maps detail is the PS_map upgraded. the_map is upgraded. flux not counts
        self.sc = source_count(A,n1,n2,Sb,Sc)
        self.exposure_map = exposure_map
        if not (self.exposure_map is None):
            self.exposure_energy_mean = np.mean(self.exposure_map, axis=0) # This is to average over the energies
            self.exposure_map_mean = np.mean(self.exposure_map) # This is to average over everything

    def config_source_count(self,max_counts,n_ps = 10000, smin = 0.001,smax = 1000., n_bins = 100000.):
        self.sc.find_count_list(max_counts,n_ps = n_ps, smin = smin,smax = smax, n_bins = n_bins)
        self.ssd.find_spatial_list(n_ps = self.sc.n_ps_real)
        # self.assign_counts() # Removed this 
       
    #@autojit 
    def assign_counts(self):
        self.counts_map_detail = np.zeros(self.ssd.npix)
        cdef int j=0
        cdef Py_ssize_t i
        for i in range(self.ssd.npix):
            if self.ssd.the_ps_map[i] > 0:
                self.counts_map_detail[i] = np.sum(self.sc.count_list[j:j+int(self.ssd.the_ps_map[i])])
                #j =j+1
                j = j+int(self.ssd.the_ps_map[i]) #check this!!

        self.counts_map = hp.ud_grade(self.counts_map_detail,self.nside,power=-2)

    def assign_counts_exposure(self):
        self.counts_map_detail = np.zeros(self.ssd.npix)
        cdef int j=0
        cdef Py_ssize_t i
        print "Length of the ssd.the_ps_map is ", len(self.ssd.the_ps_map)
        print "Length of the exposure energy mean is ", len(self.exposure_energy_mean)
        print "self.ssd.npix is ", self.ssd.npix
        for i in range(self.ssd.npix):
            if self.ssd.the_ps_map[i] > 0:
                self.counts_map_detail[i] = np.sum(self.sc.count_list[j:j+int(self.ssd.the_ps_map[i])])*self.exposure_energy_mean[i]/self.exposure_map_mean 
                j = j+int(self.ssd.the_ps_map[i]) #check this!!

        self.counts_map = hp.ud_grade(self.counts_map_detail,self.nside,power=-2)

    def cleanup(self):
        pass
        #del self.sc
        #del self.ssd

    def simplify_counts_map(self,the_mask): #requires a mask
        self.counts_map = self.counts_map*np.logical_not(the_mask)
        self.counts_map_detail = self.counts_map_detail*np.logical_not(hp.ud_grade(the_mask,self.nside_detail_factor*self.nside,power=-2) )
         
    #@autojit       
    def smear_map(self,psf,spectra_factor=1.0): #psf in sigma degrees
        print 'In smear_map: psf is ', psf
        self.non_zero_vals = np.where(self.counts_map_detail > 0)[0]
        print 'The number of PSs to smooth is ', len(self.non_zero_vals)
        self.vals_where_non_zero = np.random.poisson(spectra_factor*self.counts_map_detail[self.non_zero_vals]) #draw from poisson here
        new_non_zero = np.where(self.vals_where_non_zero > 0)[0]
        self.non_zero_vals = self.non_zero_vals[ new_non_zero ] 
        self.vals_where_non_zero = self.vals_where_non_zero [ new_non_zero ]
        self.non_zero_theta, self.non_zero_phi = hp.pix2ang(self.nside_detail_factor*self.nside, self.non_zero_vals)

        cdef Py_ssize_t i
        self.counts_map_psf = np.zeros(hp.nside2npix(self.nside))
        for i in range(len(self.non_zero_vals)):
            # if int(self.vals_where_non_zero[i]) <= 0:
            #     print 'Non-zero val actually zero val = ', self.vals_where_non_zero[i]
            # else:
            dp = draw_points(psf,int(self.vals_where_non_zero[i]),self.non_zero_theta[i],self.non_zero_phi[i], self.counts_map_psf)
            self.counts_map_psf = dp.the_map

    def smear_map_king(self,fcore,score,gcore,stail,gtail,SpE,spectra_factor=1.0): #psf in sigma degrees
        self.non_zero_vals = np.where(self.counts_map_detail > 0)[0]
        print 'The number of PSs to smooth is ', len(self.non_zero_vals)
        self.vals_where_non_zero = np.random.poisson(spectra_factor*self.counts_map_detail[self.non_zero_vals]) #draw from poisson here
        new_non_zero = np.where(self.vals_where_non_zero > 0)[0]
        self.non_zero_vals = self.non_zero_vals[ new_non_zero ]
        self.vals_where_non_zero = self.vals_where_non_zero [ new_non_zero ]
        self.non_zero_theta, self.non_zero_phi = hp.pix2ang(self.nside_detail_factor*self.nside, self.non_zero_vals)

        cdef Py_ssize_t i
        self.counts_map_psf = np.zeros(hp.nside2npix(self.nside))
        for i in range(len(self.non_zero_vals)):
            # if int(self.vals_where_non_zero[i]) <= 0:
            #     print 'Non-zero val actually zero val = ', self.vals_where_non_zero[i]
            # else:
            dp = draw_points_king(fcore,score,gcore,stail,gtail,SpE,int(self.vals_where_non_zero[i]),self.non_zero_theta[i],self.non_zero_phi[i], self.counts_map_psf)
            self.counts_map_psf = dp.the_map

        # print 'The number of counts from PSs is ', np.sum(self.vals_where_non_zero)
        # self.swp_inst = swp.smooth_gaussian_psf(psf,hp.ud_grade(self.counts_map,self.nside_smear,power=-2), mult_sigma_for_smooth = mult_sigma_for_smooth)
        # ang_vec = hp.pix2ang(self.ssd.nside,self.non_zero_vals)
        # self.counts_map_psf =  hp.ud_grade( self.swp_inst.smooth_the_pixel_by_angles(ang_vec,self.vals_where_non_zero) ,self.nside,power=-2)


        # self.counts_map_psf_2 = np.sum([self.swp_inst.smooth_the_pixel(nzv) for nzv in self.non_zero_vals],axis=0)
        
        # self.counts_map_psf = hp.smoothing(self.counts_map,sigma = 2*np.pi*psf/float(360))
                
    def plot_counts_map(self,lonra = [-10,10],latra = [-10,10],smoothed = False):
        if not smoothed:
            plot = hp.cartview(self.counts_map,lonra = lonra,latra = latra)
        else:
            plot = hp.cartview(self.counts_map_psf,lonra = lonra,latra = latra)
        return plot



class generate_ps_data_from_function():
    
    def __init__(self,the_map,S_array,dN_array,nside_detail_factor=16, exposure_map=None): #have the keyword here for using exposure.  exposure = None here
        self.nside_detail_factor=nside_detail_factor
        self.nside = hp.npix2nside(len(the_map))
        self.the_map_detail = hp.ud_grade(the_map,nside_detail_factor*self.nside,power=-2)
        self.ssd = source_spatial_dist(self.the_map_detail) #here you want to divide self.the_map_detail by the exposure map * mean of exposure map.  #the maps detail is the PS_map upgraded. the_map is upgraded. flux not counts
        self.sc = source_count_from_function(S_array,dN_array)
        self.exposure_map = exposure_map
        if not (self.exposure_map is None):
            self.exposure_energy_mean = np.mean(self.exposure_map, axis=0) # This is to average over the energies
            self.exposure_map_mean = np.mean(self.exposure_map) # This is to average over everything

    def config_source_count(self,max_counts,n_ps = 10000, smin = 0.001):
        self.sc.find_count_list(max_counts,n_ps = n_ps, smin = smin)
        self.ssd.find_spatial_list(n_ps = self.sc.n_ps_real)
        # self.assign_counts() # Removed this 
       
    #@autojit 
    def assign_counts(self):
        self.counts_map_detail = np.zeros(self.ssd.npix)
        cdef int j=0
        cdef Py_ssize_t i
        for i in range(self.ssd.npix):
            if self.ssd.the_ps_map[i] > 0:
                self.counts_map_detail[i] = np.sum(self.sc.count_list[j:j+int(self.ssd.the_ps_map[i])])
                #j =j+1
                j = j+int(self.ssd.the_ps_map[i]) #check this!!

        self.counts_map = hp.ud_grade(self.counts_map_detail,self.nside,power=-2)

    def assign_counts_exposure(self):
        self.counts_map_detail = np.zeros(self.ssd.npix)
        cdef int j=0
        cdef Py_ssize_t i
        print "Length of the ssd.the_ps_map is ", len(self.ssd.the_ps_map)
        print "Length of the exposure energy mean is ", len(self.exposure_energy_mean)
        print "self.ssd.npix is ", self.ssd.npix
        for i in range(self.ssd.npix):
            if self.ssd.the_ps_map[i] > 0:
                self.counts_map_detail[i] = np.sum(self.sc.count_list[j:j+int(self.ssd.the_ps_map[i])])*self.exposure_energy_mean[i]/self.exposure_map_mean 
                j = j+int(self.ssd.the_ps_map[i]) #check this!!

        self.counts_map = hp.ud_grade(self.counts_map_detail,self.nside,power=-2)

    def cleanup(self):
        pass
        #del self.sc
        #del self.ssd

    def simplify_counts_map(self,the_mask): #requires a mask
        self.counts_map = self.counts_map*np.logical_not(the_mask)
        self.counts_map_detail = self.counts_map_detail*np.logical_not(hp.ud_grade(the_mask,self.nside_detail_factor*self.nside,power=-2) )
         
    #@autojit       
    def smear_map(self,psf,spectra_factor=1.0): #psf in sigma degrees
        print 'In smear_map: psf is ', psf
        self.non_zero_vals = np.where(self.counts_map_detail > 0)[0]
        print 'The number of PSs to smooth is ', len(self.non_zero_vals)
        self.vals_where_non_zero = np.random.poisson(spectra_factor*self.counts_map_detail[self.non_zero_vals]) #draw from poisson here
        new_non_zero = np.where(self.vals_where_non_zero > 0)[0]
        self.non_zero_vals = self.non_zero_vals[ new_non_zero ] 
        self.vals_where_non_zero = self.vals_where_non_zero [ new_non_zero ]
        self.non_zero_theta, self.non_zero_phi = hp.pix2ang(self.nside_detail_factor*self.nside, self.non_zero_vals)

        cdef Py_ssize_t i
        self.counts_map_psf = np.zeros(hp.nside2npix(self.nside))
        for i in range(len(self.non_zero_vals)):
            # if int(self.vals_where_non_zero[i]) <= 0:
            #     print 'Non-zero val actually zero val = ', self.vals_where_non_zero[i]
            # else:
            dp = draw_points(psf,int(self.vals_where_non_zero[i]),self.non_zero_theta[i],self.non_zero_phi[i], self.counts_map_psf)
            self.counts_map_psf = dp.the_map

    def smear_map_king(self,fcore,score,gcore,stail,gtail,SpE,spectra_factor=1.0): #psf in sigma degrees
        self.non_zero_vals = np.where(self.counts_map_detail > 0)[0]
        print 'The number of PSs to smooth is ', len(self.non_zero_vals)
        self.vals_where_non_zero = np.random.poisson(spectra_factor*self.counts_map_detail[self.non_zero_vals]) #draw from poisson here
        new_non_zero = np.where(self.vals_where_non_zero > 0)[0]
        self.non_zero_vals = self.non_zero_vals[ new_non_zero ]
        self.vals_where_non_zero = self.vals_where_non_zero [ new_non_zero ]
        self.non_zero_theta, self.non_zero_phi = hp.pix2ang(self.nside_detail_factor*self.nside, self.non_zero_vals)

        cdef Py_ssize_t i
        self.counts_map_psf = np.zeros(hp.nside2npix(self.nside))
        for i in range(len(self.non_zero_vals)):
            # if int(self.vals_where_non_zero[i]) <= 0:
            #     print 'Non-zero val actually zero val = ', self.vals_where_non_zero[i]
            # else:
            dp = draw_points_king(fcore,score,gcore,stail,gtail,SpE,int(self.vals_where_non_zero[i]),self.non_zero_theta[i],self.non_zero_phi[i], self.counts_map_psf)
            self.counts_map_psf = dp.the_map
