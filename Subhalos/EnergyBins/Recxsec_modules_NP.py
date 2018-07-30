import os, sys
import copy

import argparse
import numpy as np
import iminuit
from iminuit import Minuit, describe, Struct

# NPTFit modules
sys.path.append("/tigress/somalwar/NPTFit")
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis

def makeMockData(signal, *args):
    mockData = np.array(args[0])
    for arg in args[1:]:
        mockData += np.array(arg)
    mockData = np.random.poisson(mockData)+signal
    mockData = np.round(mockData).astype(np.int32)
    return mockData

def getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_temp_ebins, bkg_temp_NP_ebins, sub_temp, flux, *args ):

    ll_ebins = []
    A0_ebins = []
    Fb_arr_ebins = []
    for ib, data_arr in enumerate(data_ebins):
        n = nptfit.NPTF(tag=tag)
        n.load_data(data_arr, exposure_ebins[ib])
        if ib==0: np.save("fake_data", data_arr)
        n.load_mask(mask)

        for temp in bkg_temp_ebins[ib]:
            n.add_template(copy.deepcopy(temp[0]), temp[1])
            n.add_poiss_model(temp[1], 'A_\mathrm{'+temp[1]+'}$', [0,20], False)
        n.add_template(copy.deepcopy(sub_temp), "subs")
        
        for temp in bkg_temp_NP_ebins[ib]:
            n.add_template(copy.deepcopy(temp[0]), temp[1])
            n.add_non_poiss_model( temp[1], 
                                   ['$A^\mathrm{ps}_\mathrm{iso}$','$n_1$','$n_2$', '$n_3$', '$S_b1$', '$S_b2'],
#                                   [[-7,-2],[2,10],[-20.,20.],[0., 2.],[0, 20],[0, 20]],
                                   [[-10,20],[2.05,5],[-20.,20.],[-1.99, 1.99],[0, 20],[0, 20]],
                                   [True,False,False, False, False, False] )

        A0 = args[ib]
        n_arr = args[len(data_ebins)+(Nb+1)*ib:len(data_ebins)+Nb+1+(Nb+1)*ib]
        Fb_arr = args[len(data_ebins)*( 1 + (Nb+1) ) + (Nb)*ib:len(data_ebins)*( 1 + (Nb+1) ) + (Nb)*ib + Nb ]
        if len(n_arr) != len(Fb_arr)+1: 
            print( "Invalid source count distribution parameters!")
            return np.nan

        if flux: A0, Fb_arr = SCDParams_Flux2Counts(A0, Fb_arr, mask, exposure_ebins[ib], sub_temp)
        print(A0, n_arr, Fb_arr)
        A0_ebins.append(A0)
        Fb_arr_ebins.append(Fb_arr)
        param_names = [ '$A_\mathrm{SCD}$' ]
        for ni in range(len(n_arr)):
            param_names.append( '$n_' + str(ni+1) + '$' )
        for Fb in range(len(Fb_arr)):
            param_names.append( '$F_{b' + str(Fb+1) + '}$' )

        fixed_params = []
        for ni, n_val in enumerate(n_arr):
            fixed_params.append( [ni+1, n_val] )

        n.add_non_poiss_model('subs',
                              param_names,
                              fixed_params = fixed_params,
                              units='counts')
        n.configure_for_scan()
        ll_ebins.append(n.ll)
    return ll_ebins, A0_ebins, Fb_arr_ebins

def SCDParams_Flux2Counts(A, Fb_arr, mask, exposure, sub_temp):
    area_mask = len(mask[~mask])/len(mask) * 4*np.pi * (180./np.pi)**2
    mean_exp = np.average(exposure[~mask])
    sub_temp_sum = np.sum(sub_temp[~mask])

    A_counts = A / mean_exp * area_mask / sub_temp_sum
    Fb_counts = []
    for Fb in Fb_arr:
        Fb_counts.append( Fb * mean_exp )
    Fb_counts = np.array(Fb_counts)

    return A_counts, Fb_counts
