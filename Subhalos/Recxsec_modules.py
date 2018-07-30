import os
import copy

import argparse
import numpy as np
import iminuit
from iminuit import Minuit, describe, Struct

# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis

def makeMockData(signal, *args):
    mockData = np.array(args[0])
    for arg in args[1:]:
        mockData += np.array(arg)
    mockData = np.random.poisson(mockData)+signal
    np.save("MC/signal", mockData)
    mockData = np.round(mockData).astype(np.int32)
    return mockData

def getNPTFitLL( data, exposure, mask, Nb, tag, bkg_temp, sub_temp, *args ):
    n = nptfit.NPTF(tag=tag)
    n.load_data(data, exposure)
    n.load_mask(mask)

    for temp in bkg_temp:
        n.add_template(copy.deepcopy(temp[0]), temp[1])
        n.add_poiss_model(temp[1], 'A_\mathrm{'+temp[1]+'}$', [0,20], False)
    n.add_template(copy.deepcopy(sub_temp), "subs")
    
    A0 = args[0]
    n_arr = args[1:Nb+2]
    Fb_arr = args[Nb+2:]
    if len(n_arr) != len(Fb_arr)+1: 
        print( "Invalid source count distribution parameters!")
        return np.nan
    
    A0, Fb_arr = SCDParams_Flux2Counts(A0, Fb_arr, mask, exposure, sub_temp)

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
    return n.ll, A0, Fb_arr

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
