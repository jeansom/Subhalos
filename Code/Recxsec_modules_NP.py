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
    mockData = np.random.poisson(mockData+signal)
    mockData = np.round(mockData).astype(np.int32)
    return mockData

def addPBkg(n, bkgt_a):
    for bkgt_at in bkgt_a:
        n.add_template(copy.deepcopy(bkgt_at[0]), bkgt_at[1], units='counts')
        n.add_poiss_model(bkgt_at[1], 'A', [0,20], False)  
    return n

def addNPBkg(n, bkgtNP_a):
    for bkgtNP_at in bkgtNP_a:
        if len(bkgtNP_at) == 0: continue
        n.add_template(copy.deepcopy(bkgtNP_at[0]), bkgtNP_at[1], units='PS')
        n.add_non_poiss_model( bkgtNP_at[1], 
                               ['$A$','$n_1$','$n_2$', '$n_3$', '$S_b1$', '$S_b2$'],
                               [[-10, -1],[2.05, 10],[-3, 3],[-10, 1.95],[0.1,100],[0.1,1]],
                               [True,False,False, False, False, False],
                                  dnds_model='specify_relative_breaks')
    return n

def addSig(n, sigt_a):
    n.add_template(copy.deepcopy(sigt_a), "subs", units='PS')
    return n

def getParaNames(Nb):
    pn_a = ['$A$']
    for ni_t in range(Nb+1):
        pn_a.append( '$n_' + str(ni_t+1) + '$' )
    for Fb_t in range(Nb):
        pn_a.append( '$F_{b' + str(Fb_t+1) + '}$' )
    return pn_a

def getFixedPara(n_a):
    fp_a = []
    for ni, n_val in enumerate(n_a):
        fp_a.append( [ni+1, n_val] )
    return fp_a

def getNPTFitLL( data_a, exp_a, mask_a, Nb, tag, bkgt_a, bkgtNP_a, sigt_a, floatSig, includeSig, *args ):

    ll_a, A0_a, Fb_a, n_a = [], [], [], []
    for ib, d_at in enumerate(data_a):
        n = nptfit.NPTF(tag=tag)
        n.load_data(d_at, exp_a[ib])
        n.load_mask(mask_a)
        
        n = addPBkg(n, bkgt_a[ib])
        n = addSig(n, sigt_a)
        n = addNPBkg(n, bkgtNP_a)
        
        n_at = args[len(data_a)+(Nb+1)*ib:len(data_a)+Nb+1+(Nb+1)*ib]
        Fb_at = args[len(data_a)*( 1 + (Nb+1) ) + (Nb)*ib:len(data_a)*( 1 + (Nb+1) ) + (Nb)*ib + Nb ]
        if len(n_at) != len(Fb_at)+1: 
            return np.nan

        A0_a.append(10**args[ib])
        Fb_at = np.array(Fb_at)
        for iF_t, Fb_t in enumerate(Fb_at):
            if iF_t != 0:
                Fb_at[iF_t] = Fb_at[iF_t-1]*Fb_t
        Fb_a.append(Fb_at.copy())

        pn_at = getParaNames(len(Fb_at))
        fp_at = getFixedPara(n_at)

        if includeSig:
            if not floatSig: 
                n.add_non_poiss_model('subs',
                                      pn_at,
                                      fixed_params = fp_at)
            else:
                n.add_non_poiss_model( 'subs', 
                                   ['$A$','$n_1$','$n_2$', '$n_3$', '$S_b1$', '$S_b2$'],
                                       [[1e-10, 1e-1],[2.05, 10],[-3, 3],[-10, 1.95],[0.1,1000],[0.1,1000]],
                                       [False,False,False, False, False, False] )

        n.configure_for_scan()
        ll_a.append(n.ll)
        n_a.append(n)
    return ll_a, A0_a, Fb_a, n_a

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
