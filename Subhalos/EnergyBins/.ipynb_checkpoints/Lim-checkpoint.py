# Tigress dirs
nptf_old_dir='/tigress/smsharma/Fermi-NPTF-exposure/'
work_dir = '/tigress/smsharma/Fermi-SmoothGalHalo/'
psf_dir='/tigress/smsharma/public/CTBCORE/psf_data/'
maps_dir='/tigress/smsharma/public/CTBCORE/'
fermi_data_dir='/tigress/smsharma/public/FermiData/'

import os, sys
import copy

import argparse
import numpy as np
import iminuit
from iminuit import Minuit, describe, Struct
from scipy.interpolate import interp1d
from scipy.integrate import quad

# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis
import pandas as pd

# Additional modules
sys.path.append('/tigress/somalwar/Fermi-NPTF-exposure')
sys.path.append('/tigress/somalwar/Fermi-NPTF-exposure/pulsars/')
import fermi.fermi_plugin as fp
sys.path.append(work_dir + '/mkDMMaps')
sys.path.append(nptf_old_dir)
import mkDMMaps

# My Modules
from Recxsec_modules import makeMockData, getNPTFitLL, SCDParams_Flux2Counts

# Global settings
nside=128
eventclass=5 # 2 (Source) or 5 (UltracleanVeto)
eventtype=0 # 0 (all), 3 (bestpsf) or 5 (top3 quartiles)
diff = 'p6' # 'p6', 'p7', 'p8'
emin = 0
emax = 39

# Load the Fermi plugin
f_global = fp.fermi_plugin(maps_dir,fermi_data_dir=fermi_data_dir,work_dir=work_dir,CTB_en_min=emin,CTB_en_max=emax+1,nside=nside,eventclass=eventclass,eventtype=eventtype,newstyle=1,data_July16=True)

# These are the energy bin edges over which the data is defined
ebins = 2*np.logspace(-1,3,41)[emin:emax+2]
my_ebins = [ .2, 2, 20, 2000 ]
my_iebins = []
for my_bin in my_ebins:
    my_iebins.append(np.argmin(np.abs(my_bin - ebins)))
my_iebins = np.unique(my_iebins)

# Load necessary templates from the plugin

f_global.add_diffuse_newstyle(comp = diff,eventclass = eventclass, eventtype = eventtype) 
f_global.add_iso()  
ps_temp = np.load('/tigress/somalwar/Subhaloes/Subhalos/ps_map.npy')
f_global.add_template_by_hand(comp='ps_model',template=ps_temp)
f_global.add_bubbles()

exposure_ebins= []
dif_ebins= []
iso_ebins= []
psc_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    fermi_exposure = np.zeros(len(f_global.CTB_exposure_maps[b]))
    dif = np.zeros(len(fermi_exposure))
    iso = np.zeros(len(fermi_exposure))
    psc = np.zeros(len(fermi_exposure))
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=(ebins[bin_ind] + ebins[bin_ind+1])/2. - (ebins[bin_ind] + ebins[bin_ind-1])/2.
        fermi_exposure += f_global.CTB_exposure_maps[bin_ind]*((ebins[bin_ind] + ebins[bin_ind+1])/2. - (ebins[bin_ind] + ebins[bin_ind-1])/2.)
        dif += f_global.template_dict[diff][bin_ind]
        iso += f_global.template_dict['iso'][bin_ind]
        psc += f_global.template_dict['ps_model'][bin_ind]
    fermi_exposure = fermi_exposure / n
    exposure_ebins.append(fermi_exposure)
    dif_ebins.append(dif)
    iso_ebins.append(iso)
    psc_ebins.append(psc)


parser = argparse.ArgumentParser(description="Signal Recovery")
parser.add_argument('-n', '--num_breaks', type=int, help='Number of breaks')
parser.add_argument('-d', '--use_data', default=False, action='store_true', help='Use real data')
parser.add_argument('-x', '--xsec_inj', type=float, help='xsec to inject')
parser.add_argument('-p', '--params_SCD', nargs='+', type=float, help='SCD Params')
parser.add_argument('-t', '--tag', type=str, help='tag for NPTFit')
parser.add_argument('-m', '--mass', type=float, help='DM mass to test')
parser.add_argument('-c', '--chi_mass', type=float, help='DM mass to inject')
parser.add_argument('-u', '--useMC', type=str, help='MC to use')
parser.add_argument('-s', '--useSubhalo', type=str, help='Subhalo MC to use')
parser.add_argument('-f', '--flux', default=False, action='store_true', help='A, Fb in terms of flux or counts, default is counts')
args = parser.parse_args()

useData = args.use_data #False
xsec_inj = args.xsec_inj #1e-22
Nb = args.num_breaks #1
SCD_params = np.array(args.params_SCD) #10**(3.53399), 10.0, 1.89914, 10**(-7.71429)
SCD_params[0] = 10**SCD_params[0]
SCD_params[Nb+2:] = 10**(SCD_params[Nb+2:])
tag = args.tag #"subs1"
mass = args.mass
mass_inj = args.chi_mass
if ".npy" in args.useMC: useMadeMC = args.useMC
else: useMadeMC = None
if ".npy" in args.useSubhalo: useSubhalo = args.useSubhalo
else: useSubhalo = None
isflux = args.flux

channel = 'b'
dNdLogx_df = pd.read_csv('/tigress/somalwar/Subhaloes/Subhalos/Data/AtProduction_gammas.dat', delim_whitespace=True)
dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(mass)))))[['Log[10,x]',channel]]
Egamma = np.array(mass*(10**dNdLogx_ann_df['Log[10,x]']))
dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
dNdE_interp = interp1d(Egamma, dNdEgamma)
PPnoxsec_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    print(ib)
    print( my_iebins[ib+1])
    ebins = [ ebins[b], ebins[my_iebins[ib+1]] ]
    if ebins[0] < mass:
        if ebins[1] < mass:
            # Whole bin is inside
            PPnoxsec_ebins.append(1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins[0], ebins[1])[0])
        else:
            # Bin only partially contained
            PPnoxsec_ebins.append(1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins[0], mass)[0])

xsec0 = 1e-22
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')

subhalo_MC = []
if useSubhalo == None: 
    for ib, b in enumerate(my_iebins[:-1]):
        fake_data_arr = []
        for fac in xsec_inj * 1e22:
            fake_data = 13.9583217*dif_ebins[ib] + 1.06289421*iso_ebins[ib] + 0.90448092*psc_ebins[ib]
            fake_data = np.random.poisson(fake_data).astype(np.float64)
            fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map0_"+str(b)+".npy")*exposure_ebins[ib]*fac
            fake_data = np.round(fake_data).astype(np.int32)
            fake_data_arr.append(fake_data)
        subhalo_MC.append(fake_data_arr)
else: 
    for ib, b in enumerate(my_iebins[:-1]):
        subhalo_MC.append(np.load(useSubhalo+str(b)+".npy") * fermi_exposure * xsec_inj/xsec0 * PPnoxsec_ebins[ib]/PPnoxsec0)
    
pscmask = np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)

bkgFac = 1.

data_ebins = []
if useData: 
    #### ADD DATA
    data = FermiData
elif useMadeMC!=None:
    for ib, b in enumerate(my_iebins):
        data_ebins.append(np.load(useMadeMC+str(b)+".npy"))
else:
    for ib, b in enumerate(my_iebins):
        data_ebins.append(makeMockData( subhalo_MC[ib], bkgFac*13.9583217*dif_ebins[ib], bkgFac*1.06289421*iso_ebins[ib], bkgFac*0.90448092*psc_ebins[ib] ))

if useMadeMC==None: np.save("/tigress/somalwar/Subhaloes/Subhalos/MC/mockdata_"+str(tag), data_ebins)
bkg_arr = []
for ib in range(len(my_iebins)):
    bkg_arr.append([ [ dif_ebins[ib], 'dif'], [iso_ebins[ib], 'iso'], [psc_ebins[ib], 'psc'] ] )
ll_ebins, A_ebins, Fb_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, subhalos, isflux, *SCD_params )

def ll_tot( d, i, p, xsec ):
    ll = 0
    for ib in range(len(my_iebins)):
        ll += ll_ebins[ib]([ d, i, p, A_ebins[ib]/((xsec_t/xsec0)*(PPnoxsec/PPnoxsec0)), *(Fb_ebins[ib]*((xsec_t/xsec0)*(PPnoxsec/PPnoxsec0))) ])
    return ll

xsec_test_arr = np.logspace(-40, -20, 101)
ll_arr = []
for xsec_t in xsec_test_arr:
    minuit_min = iminuit.Minuit(lambda d, i, p: -ll_tot([ d, i, p, xsec ]), d=13.9583217, i=1.06289421, p=0.90448092, fix_d=False, fix_i=False, fix_p=False, limit_d=(0.,100.), limit_i=(0.,15.), limit_p=(0.,15.), error_d=1e-1, error_i=1e-1, error_p=1e-1, print_level=1)
    minuit_min.migrad()
    ll_arr.append(-minuit_min.fval)
ll_arr = np.array(ll_arr)

TS_xsec_ary = 2*(ll_arr - ll_arr[0])
max_loc = np.argmax(TS_xsec_ary)
max_TS = TS_xsec_ary[max_loc]

for xi in range(max_loc, len(xsec_test_arr)):
    val = TS_xsec_ary[xi] - max_TS
    if val < -2.71:
        scale = (TS_xsec_ary[xi-1]-max_TS+2.71)/(TS_xsec_ary[xi-1]-TS_xsec_ary[xi])
        xsec_rec = xsec_test_arr[xi-1] + scale*(xsec_test_arr[xi] - xsec_test_arr[xi-1])
        break

np.savez("lim_fixed_"+str(xsec_inj) + "_" + str(useData) + "_" + str(Nb) + "_" + tag, [xsec_rec, -minuit_min.fval ], ll_arr )
