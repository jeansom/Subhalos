import sys, os

import numpy as np
from numpy.random import random_sample
import scipy
import matplotlib
import matplotlib.pyplot as plt

import healpy as hp
import mpmath as mp
from astropy.io import fits

import logging
from contextlib import contextmanager
import sys, os

from scipy import integrate, interpolate

import pulsars.masks as masks
from config.make_NFW import make_NFW
from config.config_file import config 
import config.smooth_with_psf as swp
import config.compute_PSF as CPSF
#from config.analyze_results import analyze_results as ar

from fermi.fermi_plugin import fermi_plugin as fp

from numba import jit, void, int_, double, autojit


class fake_data:
    def __init__(self):
        

        self.ps_spectra = []
        self.ps_maps = []
        self.exact_ps_maps = []


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
        #self.init_back()
        # if self.use_ps:
        #     self.psf_array=psf_array
        #     self.theta_PS = theta_PS
        #     self.ps_profile = ps_profile
        #     self.mult_sigma_for_smooth, self.nside_smear_factor, self.simplify_ps_map = mult_sigma_for_smooth, nside_smear_factor,simplify_ps_map
        #     self.Sc, self.n_ps = Sc, n_ps
        #     self.mask_total = mask_total
        #     self.init_ps()


    def make_band_mask(self,band_mask_range):
        band_mask = masks.mask_lat_band(90+band_mask_range[0],90+ band_mask_range[1], self.nside)
        return band_mask

    def make_ring_mask(self,inner,outer,lat,lng):
        ring_mask = np.logical_not(masks.mask_ring(inner, outer, lat, lng, self.nside))
        return ring_mask


    def make_mask_total(self,band_mask_range = [-30,30],mask_ring = False,lat= 90,lng = 0,inner = 0, outer= 30,ps_mask_array = 'False'):
        self.band_mask_range = band_mask_range
        self.band_mask = self.make_band_mask(self.band_mask_range)

        self.mask_ring = mask_ring
        self.lat = lat
        self.lng = lng
        self.inner = inner
        self.outer = outer
        
        if not self.mask_ring:
            self.mask_geom_total = self.band_mask #fix this!
        else:
            self.ring_mask = self.make_ring_mask(self.inner,self.outer,self.lat,self.lng)
            self.mask_geom_total = self.ring_mask + self.band_mask

        self.mask_total = self.mask_geom_total
        self.mask_total_array = np.array([self.mask_geom_total for en in self.CTB_en_bins])


    def init_back(self,A):
        self.use_ps=False
        self.A_vec = A
        self.xbg = [ np.sum(np.array([self.template_dict[A[i]][en] for i in range(len(self.A_vec))]),axis=0) for en in range(len(self.CTB_en_bins)-1) ]

    def init_ps(self,ps_profile,theta_PS,psf_array,Sc=10000.,n_ps = 10000,mult_sigma_for_smooth=5,nside_smear_factor=4,simplify_ps_map=True,nside_detail_factor=16):
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
        self.mult_sigma_for_smooth = mult_sigma_for_smooth
        self.ps_profile = ps_profile
        self.A, self.n1, self.n2 , self.Sb = theta_PS_total
        self.Sc = Sc
        self.find_counts()
        self.gpd = generate_ps_data(self.ps_profile,self.A,self.n1,self.n2,self.Sb,Sc=self.Sc,nside_detail_factor=nside_detail_factor)
        self.gpd.config_source_count(self.counts,n_ps=n_ps)
        self.gpd.assign_counts()
        if self.simplify_ps_map:
            self.gpd.simplify_counts_map(self.mask_total)

        temp_map_array = []
        temp_exact_map_array = []
        for psf,spect in map(None,psf_array,ps_spectrum):
            print 'The psf is ', psf
            self.gpd.smear_map(psf,self.mult_sigma_for_smooth,nside_smear_factor=nside_smear_factor)
            temp_map_array.append(self.gpd.counts_map_psf*spect)
            temp_exact_map_array.append(self.gpd.counts_map*spect)
        self.gpd.cleanup()
        self.ps_maps.append(temp_map_array)
        self.exact_ps_maps.append(temp_exact_map_array)


    def find_counts(self):
        self.counts = np.sum(self.ps_profile)*self.A*(self.Sb**2)*( 1/(2.-self.n2) - (1 - (self.Sb**(self.n1-2))*(self.Sc**(2-self.n1)))/(2.-self.n1))
        print 'The number of counts should be ', self.counts

    def init_model(self):
        if self.use_ps:
            self.total_ps = np.sum(self.ps_maps,axis=0) 
            self.model = self.xbg + self.total_ps
        else:   
            self.model = self.xbg

    def make_fake_data(self):
        self.init_model()
        for en in range(len(self.model)):
            mod = self.model[en]
            if len(np.where(mod < 0))>0:
                print 'Warning: mean below zero in a pixel at energy bin index ', en
                self.model[en][np.where(mod<0)]=np.zeros(len(np.where(mod<0)))
                print 'fixed the problem by setting mean to zero counts, but be careful!'
        self.fake_data = [np.random.poisson(mod) for mod in self.model]

    def save_fake_data(self,fake_data_path = 'data/fake_data/test.txt.gz'):
        # if not os.path.exists(fake_data_dir):
        #     os.mkdir(fake_data_dir)
        np.savetxt(fake_data_path, self.fake_data)

        # if self.use_ps:
        #     np.savetxt(fake_data_dir + fake_data_tag, self.fake_data)
        # else:
        #     np.savetxt(fake_data_dir + fake_data_tag, [self.fake_data,self.model])

    def save_fake_data_key(self,fake_data_key_path='data/fake_data/test_key.txt.gz'):
        self.exact_ps_maps_total = np.sum( np.array(self.exact_ps_maps),axis=0 )
        np.savetxt(fake_data_key_path,self.exact_ps_maps)


class fake_fermi_data(fp,fake_data):
    def __init__(self,*args,**kwargs):
        fp.__init__(self,*args,**kwargs)
        fake_data.__init__(self)

        self.ps_spectra = []
        self.ps_maps = []
        self.exact_ps_maps = []


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


    # def make_band_mask(self,band_mask_range):
    #     band_mask = masks.mask_lat_band(90+band_mask_range[0],90+ band_mask_range[1], self.nside)
    #     return band_mask

    # def make_ring_mask(self,inner,outer,lat,lng):
    #     ring_mask = np.logical_not(masks.mask_ring(inner, outer, lat, lng, self.nside))
    #     return ring_mask


    # def make_mask_total(self,band_mask_range = [-30,30],mask_ring = False,lat= 90,lng = 0,inner = 0, outer= 30,ps_mask_array = 'False'):
    #     self.band_mask_range = band_mask_range
    #     self.band_mask = self.make_band_mask(self.band_mask_range)

    #     self.mask_ring = mask_ring
    #     self.lat = lat
    #     self.lng = lng
    #     self.inner = inner
    #     self.outer = outer
        
    #     if not self.mask_ring:
    #         self.mask_geom_total = self.band_mask #fix this!
    #     else:
    #         self.ring_mask = self.make_ring_mask(self.inner,self.outer,self.lat,self.lng)
    #         self.mask_geom_total = self.ring_mask + self.band_mask

    #     self.mask_total = self.mask_geom_total
    #     self.mask_total_array = np.array([self.mask_geom_total for en in self.CTB_en_bins])


    # def init_back(self,A):
    #     self.use_ps=False
    #     self.A_vec = A
    #     self.xbg = [ np.sum(np.array([self.template_dict[A[i]][en] for i in range(len(self.A_vec))]),axis=0) for en in range(len(self.CTB_en_bins)-1) ]

    # def init_ps(self,ps_profile,theta_PS,psf_array,Sc=10000.,n_ps = 10000,mult_sigma_for_smooth=5,nside_smear_factor=4,simplify_ps_map=True,nside_detail_factor=16):
    #     theta_PS = list(theta_PS)
    #     Sb_total=float(np.sum(theta_PS[3:]))
    #     print 'The number of energy bins is ', len(theta_PS[3:])
    #     A_total = theta_PS[0]/Sb_total
    #     # if len(theta_PS[3:])>1:
    #     #     A_total = theta_PS[0]/Sb_total
    #     # else:
    #     #     A_total = theta_PS[0]
    #     theta_PS_total = [A_total, theta_PS[1],theta_PS[2], Sb_total]
    #     ps_spectrum=np.array(theta_PS[3:]) / Sb_total 
    #     self.ps_spectra.append( ps_spectrum )
    #     self.simplify_ps_map=simplify_ps_map
    #     self.use_ps = True
    #     self.mult_sigma_for_smooth = mult_sigma_for_smooth
    #     self.ps_profile = ps_profile
    #     self.A, self.n1, self.n2 , self.Sb = theta_PS_total
    #     self.Sc = Sc
    #     self.find_counts()
    #     self.gpd = generate_ps_data(self.ps_profile,self.A,self.n1,self.n2,self.Sb,Sc=self.Sc,nside_detail_factor=nside_detail_factor)
    #     self.gpd.config_source_count(self.counts,n_ps=n_ps)
    #     self.gpd.assign_counts()
    #     if self.simplify_ps_map:
    #         self.gpd.simplify_counts_map(self.mask_total)

    #     temp_map_array = []
    #     temp_exact_map_array = []
    #     for psf,spect in map(None,psf_array,ps_spectrum):
    #         print 'The psf is ', psf
    #         self.gpd.smear_map(psf,self.mult_sigma_for_smooth,nside_smear_factor=nside_smear_factor)
    #         temp_map_array.append(self.gpd.counts_map_psf*spect)
    #         temp_exact_map_array.append(self.gpd.counts_map*spect)
    #     self.gpd.cleanup()
    #     self.ps_maps.append(temp_map_array)
    #     self.exact_ps_maps.append(temp_exact_map_array)


    # def find_counts(self):
    #     self.counts = np.sum(self.ps_profile)*self.A*(self.Sb**2)*( 1/(2.-self.n2) - (1 - (self.Sb**(self.n1-2))*(self.Sc**(2-self.n1)))/(2.-self.n1))
    #     print 'The number of counts should be ', self.counts

    # def init_model(self):
    #     if self.use_ps:
    #         self.total_ps = np.sum(self.ps_maps,axis=0) 
    #         self.model = self.xbg + self.total_ps
    #     else:   
    #         self.model = self.xbg

    # def make_fake_data(self):
    #     self.init_model()
    #     for en in range(len(self.model)):
    #         mod = self.model[en]
    #         if len(np.where(mod < 0))>0:
    #             print 'Warning: mean below zero in a pixel at energy bin index ', en
    #             self.model[en][np.where(mod<0)]=np.zeros(len(np.where(mod<0)))
    #             print 'fixed the problem by setting mean to zero counts, but be careful!'
    #     self.fake_data = [np.random.poisson(mod) for mod in self.model]

    # def save_fake_data(self,fake_data_path = 'data/fake_data/test.txt.gz'):
    #     # if not os.path.exists(fake_data_dir):
    #     #     os.mkdir(fake_data_dir)
    #     np.savetxt(fake_data_path, self.fake_data)

    #     # if self.use_ps:
    #     #     np.savetxt(fake_data_dir + fake_data_tag, self.fake_data)
    #     # else:
    #     #     np.savetxt(fake_data_dir + fake_data_tag, [self.fake_data,self.model])

    # def save_fake_data_key(self,fake_data_key_path='data/fake_data/test_key.txt.gz'):
    #     self.exact_ps_maps_total = np.sum( np.array(self.exact_ps_maps),axis=0 )
    #     np.savetxt(fake_data_key_path,self.exact_ps_maps)




# class fake_data:
#     def __init__(self,nside,En_bins,template_dict,use_ps=False,psf_array='Null',theta_PS='Null',ps_profile='Null',mult_sigma_for_smooth=5,nside_smear_factor=4,simplify_ps_map=False,Sc=10000.,n_ps = 10000,mask_total='Null'):
#         self.nside=nside
#         self.CTB_en_bins = En_bins
#         self.CTB_en_min = self.CTB_en_bins[0]
#         self.CTB_en_max = self.CTB_en_bins[-1]
#         self.nEbins = len(self.CTB_en_bins)

#         self.ps_spectra = []
#         self.ps_maps = []
#         self.exact_ps_maps = []

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

#     def init_ps(self,ps_profile,theta_PS,psf_array,Sc=10000.,n_ps = 10000,mult_sigma_for_smooth=5,nside_smear_factor=4,simplify_ps_map=True,nside_detail_factor=16):
#         # theta_PS = list(theta_PS)
#         # Sb_total=float(np.sum(theta_PS[3:]))
#         # print 'The number of energy bins is ', len(theta_PS[3:])
#         # A_total = theta_PS[0]/Sb_total
#         # theta_PS_total = [A_total, theta_PS[1],theta_PS[2], Sb_total]

#         # ps_spectrum=np.array(theta_PS[3:]) / Sb_total
#         # self.ps_spectra.append( ps_spectrum )
#         # self.A, self.n1, self.n2 , self.Sb = theta_PS_total
#         # print self.A,self.n1,self.n2,self.Sb
#         # self.find_counts()
#         # self.gpd = generate_ps_data(self.ps_profile,self.A,self.n1,self.n2,self.Sb,self.Sc)
#         # self.gpd.config_source_count(self.counts,n_ps=self.n_ps)
#         # self.gpd.assign_counts()
#         # if self.simplify_ps_map:
#         #     self.gpd.simplify_counts_map(self.mask_total)

#         # temp_map_array = []
#         # for psf,spect in map(None,self.psf_array,ps_spectrum):
#         #     print 'The psf is ', psf
#         #     self.gpd.smear_map(psf,self.mult_sigma_for_smooth,nside_smear_factor=self.nside_smear_factor)
#         #     temp_map_array.append(self.gpd.counts_map_psf*spect)
#         # self.ps_maps.append(temp_map_array)
#         theta_PS = list(theta_PS)
#         Sb_total=float(np.sum(theta_PS[3:]))
#         print 'The number of energy bins is ', len(theta_PS[3:])
#         A_total = theta_PS[0]/Sb_total
#         theta_PS_total = [A_total, theta_PS[1],theta_PS[2], Sb_total]
#         ps_spectrum=np.array(theta_PS[3:]) / Sb_total 
#         self.ps_spectra.append( ps_spectrum )
#         self.simplify_ps_map=simplify_ps_map
#         self.use_ps = True
#         self.mult_sigma_for_smooth = mult_sigma_for_smooth
#         self.ps_profile = ps_profile
#         self.A, self.n1, self.n2 , self.Sb = theta_PS_total
#         self.Sc = Sc
#         self.find_counts()
#         self.gpd = generate_ps_data(self.ps_profile,self.A,self.n1,self.n2,self.Sb,Sc=self.Sc,nside_detail_factor=nside_detail_factor)
#         self.gpd.config_source_count(self.counts,n_ps=n_ps)
#         self.gpd.assign_counts()
#         if self.simplify_ps_map:
#             self.gpd.simplify_counts_map(self.mask_total)

#         temp_map_array = []
#         temp_exact_map_array = []
#         for psf,spect in map(None,psf_array,ps_spectrum):
#             print 'The psf is ', psf
#             self.gpd.smear_map(psf,self.mult_sigma_for_smooth,nside_smear_factor=nside_smear_factor)
#             temp_map_array.append(self.gpd.counts_map_psf*spect)
#             temp_exact_map_array.append(self.gpd.counts_map*spect)
#         self.gpd.cleanup()
#         self.ps_maps.append(temp_map_array)
#         self.exact_ps_maps.append(temp_exact_map_array)

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

class fake_data_from_file(fake_fermi_data):
    def __init__(self,name ='ben',tag = 'test',run_tag = 'test',nside = 128):

        self.name = name
        self.dict_dir = 'dict/'+tag + '/'#+ run_tag + '/'
        self.run_tag = run_tag
        self.tag = tag

        self.a = ar(self.dict_dir,self.run_tag)

        self.CTB_en_min, self.CTB_en_max = self.a.the_dict['CTB_en_min'], self.a.the_dict['CTB_en_max']
        #self.nside = self.a.the_dict['nside']
        self.nside = nside
        self.data_name = self.a.the_dict['data_name']

        fake_data.__init__(self,name=self.name,tag = self.tag,CTB_en_min = self.CTB_en_min, CTB_en_max = self.CTB_en_max, nside = self.nside,data_name = self.data_name)

        print 'The model parameters are ', self.a.the_dict['params']
        #print 'Please upload templates.'


    def config_back_from_file(self):
        A_log = self.a.return_poiss_medians()
        A = self.convert_log_list(A_log,self.a.the_dict['poiss_list_is_log_prior'])
        self.summed_templates_raw = self.a.the_dict['summed_templates_not_compressed']

        self.init_back(A,summed_templates = [hp.ud_grade(temp,self.nside,power=-2) for temp in self.summed_templates_raw])

    def config_ps_from_file(self,temp_number,psf = 'False',mult_sigma_for_smooth = 5,Sc = 100000.,n_ps = 100000,simplify_ps_map = False):
        self.simplify_ps_map = simplify_ps_map
        self.Sc = Sc
        a = self.a

        self.ps_template = self.summed_templates[temp_number]
        if psf == 'False':
            self.psf = CPSF.main(a.the_dict['CTB_en_bins'], a.the_dict['spect'], self.nside, a.the_dict['psf_dir'], just_sigma = True)
        else:
            self.psf = psf

        self.log_theta_ps = [a.the_dict['s']['marginals'][a.the_dict['n_poiss']+i]['median'] for i in range(0,a.the_dict['n_non_poiss'])]


        self.theta_ps = self.convert_log_list(self.log_theta_ps,a.the_dict['non_poiss_list_is_log_prior_uncompressed'][0])

        self.init_ps(self.ps_template,self.theta_ps,psf = self.psf,Sc=self.Sc,n_ps = n_ps,mult_sigma_for_smooth = mult_sigma_for_smooth)


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
        probabilities = self.probabilities/np.sum(self.probabilities)
        bins = np.add.accumulate(probabilities)
        self.x_vec = self.x_array[np.digitize(random_sample(self.n_ps), bins)]


class source_count(random):
    def __init__(self,A,n1,n2,Sb,Sc):
        self.A, self.n1,self.n2,self.Sb,self.Sc = float(A),float(n1),float(n2),float(Sb),float(Sc)
        random.__init__(self,self.dnds)
    
    def dnds(self,s):
        if s < self.Sb:
            return self.A*(s/self.Sb)**-self.n2
        elif s > self.Sb and s < self.Sc:
            return self.A*(s/self.Sb)**-self.n1
        else:
            return 0
        
    def find_count_list(self,max_counts,n_ps = 1000, smin = 0.001,smax = 1000., n_bins = 1000.):
        self.draw_x_vec(n_ps,smin,smax,n_bins)
        self.i_stop = next(i for i in range(n_ps) if  np.sum(self.x_vec[0:i]) > max_counts)
        self.count_list = self.x_vec[0:self.i_stop]
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
    
    def find_spatial_list(self,n_ps = 1000):
        self.draw_x_vec(n_ps,0,self.npix-1,self.npix)
        self.pos_vec = np.vectorize(int)(self.x_vec)
        self.the_ps_map = np.histogram(self.pos_vec,bins=np.arange(0,self.npix+1))[0]
        
    def plot_spatial_dist(self,lonra = [-10,10],latra = [-10,10]):
        plot = hp.cartview(self.the_ps_map,lonra = lonra,latra = latra)
        return plot       



class generate_ps_data():
    
    def __init__(self,the_map,A,n1,n2,Sb,Sc=10000.,nside_detail_factor=16):
        self.nside_detail_factor=nside_detail_factor
        self.nside = hp.npix2nside(len(the_map))
        self.the_map_detail = hp.ud_grade(the_map,nside_detail_factor*self.nside,power=-2)
        self.ssd = source_spatial_dist(self.the_map_detail)
        self.sc = source_count(A,n1,n2,Sb,Sc)
        
    def config_source_count(self,max_counts,n_ps = 1000, smin = 0.001,smax = 1000., n_bins = 1000.):
        self.sc.find_count_list(max_counts,n_ps = n_ps, smin = smin,smax = smax, n_bins = n_bins)
        self.ssd.find_spatial_list(n_ps = self.sc.n_ps_real)
        self.assign_counts()
       
    @autojit 
    def assign_counts(self):
        self.counts_map_detail = np.zeros(self.ssd.npix)
        j=0
        for i in range(self.ssd.npix):
            if self.ssd.the_ps_map[i] > 0:
                self.counts_map_detail[i] = np.sum(self.sc.count_list[j:j+int(self.ssd.the_ps_map[i])])
                #j =j+1
                j = j+int(self.ssd.the_ps_map[i]) #check this!!

        self.counts_map = hp.ud_grade(self.counts_map_detail,self.nside,power=-2)

    def cleanup(self):
        pass
        #del self.sc
        #del self.ssd

    def simplify_counts_map(self,the_mask): #requires a mask
        self.counts_map = self.counts_map*np.logical_not(the_mask)
        self.counts_map_detail = self.counts_map*np.logical_not(hp.ud_grade(the_mask,self.nside_detail_factor*self.nside,power=-2) )
         
    @autojit       
    def smear_map(self,psf,mult_sigma_for_smooth,nside_smear_factor=4): #psf in sigma degrees
        self.nside_smear=nside_smear_factor*self.nside
        self.non_zero_vals = np.where(self.counts_map_detail > 0)[0]
        print 'The number of PSs to smooth is ', len(self.non_zero_vals)
        self.vals_where_non_zero = self.counts_map_detail[self.non_zero_vals]
        print 'The number of counts from PSs is ', np.sum(self.vals_where_non_zero)
        self.swp_inst = swp.smooth_gaussian_psf(psf,hp.ud_grade(self.counts_map,self.nside_smear,power=-2), mult_sigma_for_smooth = mult_sigma_for_smooth)
        ang_vec = hp.pix2ang(self.ssd.nside,self.non_zero_vals)
        self.counts_map_psf =  hp.ud_grade( self.swp_inst.smooth_the_pixel_by_angles(ang_vec,self.vals_where_non_zero) ,self.nside,power=-2)
        # self.counts_map_psf_2 = np.sum([self.swp_inst.smooth_the_pixel(nzv) for nzv in self.non_zero_vals],axis=0)
        
        # self.counts_map_psf = hp.smoothing(self.counts_map,sigma = 2*np.pi*psf/float(360))
                
    def plot_counts_map(self,lonra = [-10,10],latra = [-10,10],smoothed = False):
        if not smoothed:
            plot = hp.cartview(self.counts_map,lonra = lonra,latra = latra)
        else:
            plot = hp.cartview(self.counts_map_psf,lonra = lonra,latra = latra)
        return plot