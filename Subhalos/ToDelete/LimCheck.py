import os
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

# My Modules
from Recxsec_modules import makeMockData, getNPTFitLL, SCDParams_Flux2Counts

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

ebins = [2,6.32455532]
PPnoxsec0 = 2.97868399e-05
channel = 'b'
dNdLogx_df = pd.read_csv('Data/AtProduction_gammas.dat', delim_whitespace=True)
dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(mass)))))[['Log[10,x]',channel]]
Egamma = np.array(mass*(10**dNdLogx_ann_df['Log[10,x]']))
dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
dNdE_interp = interp1d(Egamma, dNdEgamma)
if ebins[0] < mass:
    if ebins[1] < mass:
        # Whole bin is inside
        PPnoxsec = 1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins[0], ebins[1])[0]
    else:
        # Bin only partially contained
        PPnoxsec = 1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins[0], mass)[0]

xsec0 = 1e-22
FermiData = np.load('fermi_data/fermidata_counts.npy').astype(np.int32)
fermi_exposure = np.load("EnergyBins/exposure_ebins.npy") #np.load('fermi_data/fermidata_exposure.npy')
dif = np.load("EnergyBins/dif_ebins.npy") #np.load('fermi_data/template_dif.npy')
iso = np.load("EnergyBins/iso_ebins.npy") #np.load('fermi_data/template_iso.npy')
psc = np.load("EnergyBins/psc_ebins.npy") #np.load('fermi_data/template_psc.npy')
subhalos = np.load('MC/EinastoTemplate2.npy')
if useSubhalo == None: subhalo_MC = np.load("MC/subhalo_flux_map2.npy") * fermi_exposure * xsec_inj/xsec0 * PPnoxsec/PPnoxsec0
else: 
    subhalo_MC = np.load(useSubhalo) * fermi_exposure * xsec_inj/xsec0 #* PPnoxsec/PPnoxsec0
    
pscmask = np.array(np.load('fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)

bkgFac = 1.

if useData: 
    print("here")
    data = FermiData
elif useMadeMC!=None:
    data = np.load(useMadeMC)
else: 
    data = makeMockData( subhalo_MC, bkgFac*0.9442*dif, bkgFac*0.5767*iso, bkgFac*0.5368*psc )

if useMadeMC==None: np.save("MC/mockdata_"+str(tag), data)
print(SCD_params)
ll, _, _ = getNPTFitLL( data, fermi_exposure, mask, Nb, tag, [ [dif, 'dif'], [iso, 'iso'] , [psc, 'psc']], subhalos, *SCD_params )
A = SCD_params[0]
Fb = SCD_params[Nb+2:]
print(A, Fb)
xsec_test_arr = np.logspace(-40, -20, 101)
ll_arr = []
for xsec_t in xsec_test_arr:
    minuit_min = iminuit.Minuit(lambda d, i, p: -ll([ d, i, p, A/((xsec_t/xsec0)), *(Fb*((xsec_t/xsec0))) ]), d=0.1, i=0.1, p=0.1, fix_d=False, fix_i=False, fix_p=False, limit_d=(0.,20.), limit_i=(0.,3.), limit_p=(0.,3.), error_d=1e-1, error_i=1e-1, error_p=1e-1, print_level=1)
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
print(xsec_rec, minuit_min.fval)