from __future__ import division
from __future__ import print_function

from timeit import default_timer as timer

# Tigress dirs
import os, sys
import copy

import argparse
import numpy as np
import iminuit
from iminuit import Minuit, describe, Struct
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize
from mpi4py import MPI
# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import psf_correction as pc # module for determining the PSF correction
from NPTFit import dnds_analysis
import pandas as pd
import healpy as hp

# My Modules
from Recxsec_modules_NP import makeMockData, getNPTFitLL, SCDParams_Flux2Counts

comm = MPI.COMM_WORLD

ebins = 2*np.logspace(-1,3,41)[0:41]
my_iebins = [10, 15]

parser = argparse.ArgumentParser(description="Signal Recovery")
parser.add_argument('-x', '--xsec_inj', type=float, help='xsec to inject')
parser.add_argument('-t', '--tag', type=str, help='tag for NPTFit')
parser.add_argument('-u', '--useMC', type=str, help='MC to use')
parser.add_argument('-s', '--useSubhalo', type=str, help='Subhalo MC to use')
parser.add_argument('-r', '--trial', type=float, help='trial number')
args = parser.parse_args()

xsec_inj = args.xsec_inj
Nb = 2


SCD_params_arr = [
    np.array([ -10.0 ,  8.918666049999999 , 2.334626615 , 1.62376057 , 1000,  0.01000002 ]),
    np.array([ -8.655889102668796 ,  12.79694988 , 2.189601905 , 1.482223355 , 1000,  0.011324606000000001 ]),
    np.array([ -8.451611123920127 ,  12.94986755 , 2.11153209 , 1.45667173 , 1000,  0.01047918145 ]),
    np.array([ -8.235280188032446 ,  11.87094735 , 2.05262732 , 1.4126969950000001 , 1000,  0.01 ]),
    np.array([ -8.003616831732277 ,  11.67368141 , 1.98769382 , 1.36548695 , 1000,  0.01 ]),
    np.array([ -7.870585981396807 ,  11.848506295 , 1.9675991449999999 , 1.32429562 , 1000,  0.010946066 ]),
    np.array([ -7.693274080852085 ,  12.0848035 , 1.92552477 , 1.2629223299999999 , 1000,  0.010361012900000001 ]),
    np.array([ -7.427431893474997 ,  13.890991245 , 1.83032539 , 1.200661475 , 1000,  0.01 ]),
    np.array([ -7.257030010846912 ,  12.70767685 , 1.7809880900000001 , 1.13341533 , 1000,  0.01 ]),
    np.array([ -6.928962697875536 ,  11.92603345 , 1.64328917 , 1.0507513249999998 , 1000,  0.01000001855 ]),
    np.array([ -6.609144077870798 ,  14.85208855 , 1.4829622649999998 , 1.0001286600000001 , 1000,  0.0100000004 ]),
    np.array([ -6.367519382387965 ,  14.995849 , 1.36340474 , 0.9131787060000001 , 1000,  0.01 ]),
    np.array([ -5.993621982143028 ,  14.999999955 , 1.06407731 , 1.0983020350000001 , 1000,  0.1 ]),
    np.array([ -5.87944671 ,  14.99999995 , 0.97577885 , 1.04414602 , 1000,  0.09999999 ]),
    np.array([ -6.00859495 ,  14.99999986 , 1.19288296 , 0.49007574 , 1000,  0.01153589 ]),
    np.array([ -5.997670137587651 ,  14.99999985 , 1.2080226 , 0.1650115385 , 1000,  0.0100001564 ]),
    np.array([ -5.981396075211702 ,  14.99999999 , 1.228689325 , -0.128016612 , 1000,  0.0121407496 ]),
    np.array([ -5.911098630495233 ,  14.99999995 , 1.19693577 , -0.41432553549999995 , 1000,  0.01406597625 ]),
    np.array([ -5.80323133 ,  15.0 , 1.10845323 , -0.522429949 , 1000,  0.01665154 ]),
    np.array([ -5.708220442627556 ,  15.0 , 1.0358376150000002 , -0.694212975 , 1000,  0.019946045 ]),
    np.array([ -5.68974574080091 ,  14.999999944999999 , 1.02092364 , -0.5892758600000001 , 1000,  0.031545995 ]),
    np.array([ -5.647179654336042 ,  15.0 , 0.992328665 , -0.677919175 , 1000,  0.038985535 ]),
    np.array([ -5.632786144736857 ,  15.0 , 0.975935 , -0.59362547 , 1000,  0.05784366 ]),
    np.array([ -5.55090306856425 ,  15.0 , 0.813295895 , -0.943702475 , 1000,  0.05458518 ]),
    np.array([ -5.416102414292078 ,  15.0 , 0.47981453 , -1.2250126749999999 , 1000,  0.049441835 ]),
    np.array([ -5.450951354200434 ,  15.0 , 0.454910565 , -1.42349566 , 1000,  0.065905415 ]),
    np.array([ -5.441921226182879 ,  15.0 , 0.28376865500000004 , -1.6470874800000002 , 1000,  0.07162948499999999 ]),
    np.array([ -5.50534861140303 ,  15.0 , 0.25559853499999996 , -2.01006365 , 1000,  0.09686943 ]),
    np.array([ -5.580194706341196 ,  15.0 , 0.20392323499999998 , -2.96682985 , 1000,  0.1 ]),
    np.array([ -5.650007945964724 ,  15.0 , -0.005551337985 , -5.63165199 , 1000,  0.1 ]),
    np.array([ -5.746169438591642 ,  15.0 , -0.20539378 , -9.89932293 , 1000,  0.1 ]),
    np.array([ -5.910576031061827 ,  15.0 , -0.262683405 , -10.0 , 1000,  0.1 ]),
    np.array([ -6.067513642028862 ,  15.0 , -0.447246535 , -10.0 , 1000,  0.1 ]),
    np.array([ -6.236994066699796 ,  15.0 , -1.024878625 , -10.0 , 1000,  0.1 ]),
    np.array([ -6.500009798436643 ,  14.999255654999999 , -1.2212995549999999 , -9.999999795 , 1000,  0.1 ]),
    np.array([ -6.747695507430577 ,  13.731435300000001 , -2.2765198 , -9.989832965000002 , 1000,  0.1 ]),
    np.array([ -7.145495674812008 ,  13.1683409 , -2.951610445 , -9.69428019 , 1000,  0.1 ]),
    np.array([ -7.781019793739187 ,  14.856469 , -2.999973745 , -7.155506605 , 1000,  0.1 ])
] # 1000 GeV

'''
SCD_params_arr = [
    np.array([ -7.43207655 ,  13.77909273 , 1.83261239 , 1.18826229 , 1000,  0.01 ]),
    np.array([ -6.102498659146512 ,  14.999999835 , 1.2759519099999999 , 0.440868774 , 1000,  0.010736085 ]),
    np.array([ -6.018748812572563 ,  14.999999955 , 1.23014541 , 0.194026682 , 1000,  0.01000089115 ]),
    np.array([ -6.009473529033173 ,  15.0 , 1.253986005 , -0.0545406691 , 1000,  0.01141911235 ]),
    np.array([ -5.917287325887452 ,  15.0 , 1.19764037 , -0.37117137099999997 , 1000,  0.01336409075 ]),
    np.array([ -5.871945314864061 ,  15.0 , 1.1787895800000001 , -0.39794081 , 1000,  0.019071375 ]),
    np.array([ -5.717334580350461 ,  15.0 , 1.030699775 , -0.81588859 , 1000,  0.01613587 ]),
    np.array([ -5.6843982429043 ,  15.0 , 1.0101737700000002 , -0.657012975 , 1000,  0.02500614 ]),
    np.array([ -5.660410426569447 ,  15.0 , 1.011927225 , -0.604420805 , 1000,  0.037293425 ]),
    np.array([ -5.644567685361244 ,  15.0 , 1.00414854 , -0.5979504950000001 , 1000,  0.05341414 ]),
    np.array([ -5.5469828174072795 ,  15.0 , 0.81559759 , -0.82738367 , 1000,  0.05498539 ]),
    np.array([ -5.412407565298666 ,  15.0 , 0.50081681 , -1.486777925 , 1000,  0.0449602 ]),
    np.array([ -5.4763610277311585 ,  15.0 , 0.56705287 , -1.205330595 , 1000,  0.07047798499999999 ]),
    np.array([ -5.43827513983619 ,  15.0 , 0.32331943500000004 , -1.530448335 , 1000,  0.075962415 ]),
    np.array([ -5.478182910658986 ,  15.0 , 0.24343109 , -1.9757856550000001 , 1000,  0.08783439500000001 ]),
    np.array([ -5.567467444641143 ,  15.0 , 0.232812415 , -2.5840283250000002 , 1000,  0.1 ]),
    np.array([ -5.648557840690043 ,  15.0 , 0.142911445 , -5.34076466 , 1000,  0.1 ]),
    np.array([ -5.6637882504012484 ,  15.0 , -0.29999110500000004 , -8.52077317 , 1000,  0.1 ]),
    np.array([ -5.8517003710885165 ,  15.0 , -0.21878478 , -10.0 , 1000,  0.1 ]),
    np.array([ -6.040956966956282 ,  15.0 , -0.34495924499999997 , -10.0 , 1000,  0.1 ]),
    np.array([ -6.1665527272812675 ,  15.0 , -0.8422002150000001 , -10.0 , 1000,  0.1 ]),
    np.array([ -6.390334988646612 ,  14.99979338 , -1.21238506 , -10.0 , 1000,  0.1 ]),
    np.array([ -6.694458636946279 ,  14.984949705 , -1.6263217399999998 , -9.998616649999999 , 1000,  0.1 ]),
    np.array([ -7.025339669702717 ,  13.27337559 , -2.9371325 , -9.965098959999999 , 1000,  0.1 ]),
    np.array([ -7.488718945170558 ,  14.14565964 , -2.98653352 , -7.6872206 , 1000,  0.09999995 ])
] # 10 GeV
'''
'''
SCD_params_arr = [
    np.array([ -7.509501845509425, 11.91446501 , 1.1966280450000002 , 1.83436946 , 1.2721610399999999 , 1000,  0.33755266100000003 , 0.015552729999999999 ]),
    np.array([ -6.5813123250162215, 11.26125553 , 1.2119765249999999 , 1.57870279 , 0.595372132 , 1000,  0.47692904 , 0.017726025 ]),
    np.array([ -6.45235477, 12.3504826 , 1.18025594 , 1.51900557 , 0.39656674 , 1000,  0.550782153 , 0.02098431 ]),
    np.array([ -6.328337255407445, 11.887592354999999 , 1.189628605 , 1.4353811049999998 , 0.1176336855 , 1000,  0.59879997 , 0.018760045000000003 ]),
    np.array([ -6.28817476, 10.84965514 , 1.21614217 , 1.42257736 , 0.04969886 , 1000,  0.34018691 , 0.03429433 ]),
    np.array([ -6.160973945447159, 12.57782846 , 1.2027626200000001 , 1.3621168350000001 , -0.110964805 , 1000,  0.57065407 , 0.033121015000000004 ]),
    np.array([ -6.051831693636209, 14.99967004 , 1.15028139 , 1.19275972 , -0.482896538 , 1000,  0.085238005 , 0.08245048499999999 ]),
    np.array([ -5.987475025443566, 13.04779941 , 1.2487577 , 0.944640995 , -0.7684728999999999 , 1000,  0.086888625 , 0.11543937 ]),
    np.array([ -5.90539896818301, 12.753480235 , 1.231226425 , 0.76286835 , -1.05592066 , 1000,  0.0787871 , 0.14107652999999998 ]),
    np.array([ -5.8117029691320505, 12.537491335 , 1.1876255 , 0.46423022 , -1.58693416 , 1000,  0.06298016000000001 , 0.17266988 ]),
    np.array([ -5.777088853728099, 12.706262395 , 1.184882795 , 0.41761857 , -1.87619383 , 1000,  0.08187077000000001 , 0.141017255 ]),
    np.array([ -5.70326658, 14.9999921 , 0.94282868 , 0.3945169 , -1.05628122 , 1000,  0.11255024 , 0.16027463 ]),
    np.array([ -5.649876028917738, 15.0 , 0.697030085 , 0.535581265 , -1.393816585 , 1000,  0.23277658 , 0.11977394499999999 ]),
    np.array([ -5.466763451982291, 15.0 , 0.62334924 , -0.47382417099999996 , -1.7714318599999999 , 1000,  0.05352175 , 0.21141948500000002 ]),
    np.array([ -5.38610682, 15.0 , 0.36829902 , -1.44938633 , -1.33200428 , 1000,  0.04362785 , 0.08292938 ]),
    np.array([ -5.40454998, 15.0 , 0.30718182 , -1.39235539 , -1.26551823 , 1000,  0.05663381 , 0.06139623 ]),
    np.array([ -5.489557214529957, 15.0 , 0.346774121 , -1.31649316 , -0.341588057 , 1000,  0.0902879845 , 0.015287115 ]),
    np.array([ -5.537358, 15.0 , 0.314887509 , -1.69966683 , -0.0750432776 , 1000,  0.122328293 , 0.01652118 ]),
    np.array([ -5.58692574, 15.0 , 0.13492537 , -2.27605755 , -0.14989929 , 1000,  0.12858205 , 0.01509104 ]),
    np.array([ -5.624596917500925, 15.0 , -0.209988588 , -2.902109875 , 0.09489837385 , 1000,  0.137520995 , 0.0119626079 ]),
    np.array([ -5.77866704, 15.0 , -0.107720847 , -2.99999568 , 0.0980182102 , 1000,  0.203180359 , 0.0100007984 ]),
    np.array([ -5.911987263220273, 15.0 , -0.3958950035 , -2.99999989 , 0.0982093107 , 1000,  0.24809970450000002 , 0.010393228550000001 ]),
    np.array([ -6.07005179, 15.0 , -0.74293848 , -3.0 , 0.09871584 , 1000,  0.292613359 , 0.0103857252 ]),
    np.array([ -6.271000506994744, 14.9991327 , -1.1652109099999999 , -3.0 , 0.09887458404999999 , 1000,  0.33798762 , 0.0102786588 ]),
    np.array([ -6.56963046, 14.98627065 , -1.4722051 , -2.00957214 , 0.0991160607 , 1000,  0.0134573024 , 0.0100040019 ]),
    np.array([ -6.82670033103187, 14.9837442 , -2.7645761699999998 , -2.0704961500000003 , 0.09888223674999999 , 1000,  0.01004293065 , 0.010000008349999999 ]),
    np.array([ -7.42312468, 14.0607133 , -2.85537665 , -1.79726539 , 0.0990089431 , 1000,  0.0100014031 , 0.010000002 ])
]
'''
SCD_params_xsec = np.array([
    1e-22,
    1e-21 ,
    1.4125375446227497e-21 ,
    1.9952623149688665e-21 ,
    2.818382931264449e-21 ,
    3.981071705534986e-21 ,
    5.6234132519034906e-21 ,
    7.943282347242789e-21 ,
    1.1220184543019562e-20 ,
    1.5848931924611108e-20 ,
    2.238721138568347e-20 ,
    3.162277660168379e-20 ,
    4.4668359215096164e-20 ,
    6.309573444801891e-20 ,
    8.912509381337441e-20 ,
    1.2589254117941713e-19 ,
    1.7782794100389227e-19 ,
    2.5118864315095717e-19 ,
    3.548133892335731e-19 ,
    5.011872336272715e-19 ,
    7.079457843841402e-19 ,
    1e-18 ,
    1.4125375446227497e-18 ,
    1.9952623149688666e-18 ,
    2.818382931264449e-18 ,
    3.981071705534985e-18 ,
    5.623413251903491e-18 ,
    7.94328234724279e-18 ,
    1.1220184543019561e-17 ,
    1.584893192461111e-17 ,
    2.238721138568347e-17 ,
    3.1622776601683796e-17 ,
    4.466835921509616e-17 ,
    6.309573444801891e-17 ,
    8.912509381337441e-17 ,
    1.2589254117941662e-16 ,
    1.7782794100389228e-16 ,
    2.5118864315095717e-16 ,
    3.5481338923357457e-16 ,
    5.011872336272715e-16 ,
    7.079457843841373e-16
])

blazar_SCD = np.array([-4.733327518453883, 2.4461263, 1.77293727, 1.48618555, 60.55017686, 0.1 ])
tag = args.tag 
mass = 1000
mass_inj = 100
#PPnoxsec0_ebins = np.array([0.0000238640102822424, 0.00000592280390900841]) # 100 GeV
trial = int(args.trial)
if "d" in args.useMC: useMadeMC = args.useMC
else: useMadeMC = None
if "sub" in args.useSubhalo: useSubhalo = args.useSubhalo
else: useSubhalo = None

exposure_ebins= []
dif_ebins= []
iso_ebins= []
psc_ebins = []
fermi_data_ebins = []
blazars_ebins = []

for ib, b in enumerate(my_iebins[:-1]):
    fermi_exposure = np.zeros(hp.nside2npix(128))
    dif = np.zeros(len(fermi_exposure))
    iso = np.zeros(len(fermi_exposure))
    psc = np.zeros(len(fermi_exposure))
    data = np.zeros(len(fermi_exposure))
    n = 0
    for bin_ind in range(b, my_iebins[ib+1]):
        n+=1
        fermi_exposure += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/exposure'+str(bin_ind)+'.npy')
        dif += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/dif'+str(bin_ind)+'.npy')
        iso += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/iso'+str(bin_ind)+'.npy')
        psc += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/psc'+str(bin_ind)+'.npy')
        data += np.load('/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/maps/data'+str(bin_ind)+'.npy')
    fermi_exposure = fermi_exposure / n
    dif_ebins.append(dif)
    iso_ebins.append(iso)
    psc_ebins.append(psc)
    fermi_data_ebins.append(data.astype(np.int32))
    exposure_ebins.append(fermi_exposure)
    blazars_ebins.append(np.load("/tigress/somalwar/Subhaloes/Subhalos/EnergyBins/blazarMC/blazar_map_test_"+str(b)+"_"+str(my_iebins[ib+1])+"_"+str(trial)+".npy")*fermi_exposure)

channel = 'b'
dNdLogx_df = pd.read_csv('/tigress/somalwar/Subhaloes/Subhalos/Data/AtProduction_gammas.dat', delim_whitespace=True)
dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(mass)))))[['Log[10,x]',channel]]
Egamma = np.array(mass*(10**dNdLogx_ann_df['Log[10,x]']))
dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
dNdE_interp = interp1d(Egamma, dNdEgamma)
PPnoxsec_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    ebins_temp = [ ebins[b], ebins[my_iebins[ib+1]] ]
    if ebins_temp[0] < mass:
        if ebins_temp[1] < mass:
            # Whole bin is inside
            PPnoxsec_ebins.append(1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], ebins_temp[1])[0])
        else:
            # Bin only partially contained
            PPnoxsec_ebins.append(1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], mass)[0])
    else: PPnoxsec_ebins.append(0)

PPnoxsec0_ebins = []
for ib, b in enumerate(my_iebins[:-1]):
    ebins_temp = [ ebins[b], ebins[my_iebins[ib+1]] ]
    if ebins_temp[0] < mass_inj:
        if ebins_temp[1] < mass_inj:
            # Whole bin is inside
            PPnoxsec0_ebins.append(1.0/(8*np.pi*mass_inj**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], ebins_temp[1])[0])
        else:
            # Bin only partially contained
            PPnoxsec0_ebins.append(1.0/(8*np.pi*mass_inj**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], mass)[0])
    else: PPnoxsec0_ebins.append(0)

xsec0 = 1e-22
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')
subhalos = subhalos/np.mean(subhalos)

subhalo_MC = []
if useSubhalo == None: 
    for ib, b in enumerate(my_iebins[:-1]):
        fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map_NFW_"+str(b)+"-"+str(my_iebins[ib+1])+"_"+str(SCD_params_xsec[np.argmin(np.abs(SCD_params_xsec - xsec_inj))])+".npy")*np.mean(exposure_ebins[ib])*xsec_inj/SCD_params_xsec[np.argmin(np.abs(SCD_params_xsec - xsec_inj))]
        fake_data = np.round(np.random.poisson(fake_data)).astype(np.int32)
        subhalo_MC.append(fake_data)
        subhalo_MC[-1][subhalo_MC[-1] > 1000] = 0
else: 
    for ib, b in enumerate(my_iebins[:-1]):
        if mass_inj != 100: 
            subhalo_MC.append(np.load(useSubhalo+str(b)+"-"+str(my_iebins[ib+1])+"_1e-22.npy")*np.mean(exposure_ebins[ib])*xsec_inj/xsec0)
        else: subhalo_MC.append(np.load(useSubhalo+str(b)+"-"+str(my_iebins[ib+1])+".npy")*np.mean(exposure_ebins[ib])*xsec_inj/xsec0*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])
        subhalo_MC[-1][subhalo_MC[-1] > 1000] = 0

pscmask = np.array(np.load('/tigress/somalwar/Subhaloes/Subhalos/fermi_data/fermidata_pscmask.npy'), dtype=bool)
mask = cm.make_mask_total(band_mask = True, band_mask_range = 5, mask_ring = True, inner = 20, outer = 180, custom_mask = pscmask)

data_ebins = []
if comm.rank == 0:
    for ib, b in enumerate(my_iebins[:-1]):
        n_bkg = nptfit.NPTF(tag='norm')
        n_bkg.load_data(fermi_data_ebins[ib].copy(), exposure_ebins[ib].copy())
        n_bkg.load_mask(mask)
        
        n_bkg.add_template(dif_ebins[ib].copy(), 'dif')
        n_bkg.add_template(iso_ebins[ib].copy(), 'iso')
        n_bkg.add_template(psc_ebins[ib].copy(), 'psc')
        
        n_bkg.add_poiss_model('dif', '$A_\mathrm{dif}$', [0,20], False)
        n_bkg.add_poiss_model('iso', '$A_\mathrm{iso}$', [0,20], False)
        n_bkg.add_poiss_model('psc', '$A_\mathrm{psc}$', [0,20], False)

        n_bkg.configure_for_scan()

        bkg_min = minimize( lambda args: -n_bkg.ll([*args]), 
                            [ 0.89, 5, 0.03795109 ], method="SLSQP", bounds = [ [0,10], [0,10], [0,10] ], options={'ftol':1e-15, 'eps':1e-10, 'maxiter':5000, 'disp':True} )
        print(bkg_min.x)
        data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib], bkg_min.x[0]*dif_ebins[ib], bkg_min.x[1]*iso_ebins[ib] ))
        np.save("MPITemp/fake_data"+str(ib)+"_"+tag, data_ebins[-1])
comm.Barrier()
if comm.rank != 0:
    for ib, b in enumerate(my_iebins[:-1]):
        data_ebins.append(np.load("MPITemp/fake_data"+str(ib)+"_"+tag+".npy"))

bkg_arr = []
for ib in range(len(my_iebins)-1):
#    bkg_arr.append([ [ dif_ebins[ib], 'dif'], [iso_ebins[ib], 'iso'], [psc_ebins[ib], 'psc'] ] )
    bkg_arr.append([ [ dif_ebins[ib], 'dif'], [iso_ebins[ib], 'iso'] ])
#    bkg_arr.append([])

bkg_arr_np = [[[np.ones(len(blazars_ebins[ib])), 'blaz']], [[np.ones(len(blazars_ebins[ib])), 'blaz']]]

ll_ebins_xsec = []
A_ebins_xsec = []
Fb_ebins_xsec = []
Fb_ebins_ub_xsec= []
Fb_ebins_lb_xsec= []
n_ebins_xsec = []
for ix in range(len(SCD_params_arr)):
    ll_ebins, A_ebins, Fb_ebins, n_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, bkg_arr_np, subhalos, False, False, True, *SCD_params_arr[ix] )
    ll_ebins_xsec.append(ll_ebins)
    A_ebins_xsec.append(A_ebins)
    Fb_ebins_xsec.append(Fb_ebins)
    n_ebins_xsec.append(n_ebins)

#    _, _, Fb_ebins_ub, _ = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, bkg_arr_np, subhalos, False, False, True, *SCD_params_ub[ix] )
#    Fb_ebins_ub_xsec.append(Fb_ebins_ub)

#    _, _, Fb_ebins_lb, _ = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, bkg_arr_np, subhalos, False, False, True, *SCD_params_lb[ix] )
#    Fb_ebins_lb_xsec.append(Fb_ebins_lb)

ll_arr = []
SCDb_arr_ebins = []
def ll_func( xsec_t, ix, A_dif, A_iso, A_psc, Ab, n1b, n2b, n3b, Fb1b, Fb2b, Ab_sig, Fb1_sig, Fb2_sig ): 
    return -ll_ebins_xsec[ix][ib]([ A_dif, A_iso, Ab, n1b, n2b, n3b, Fb1b, Fb2b, Ab_sig, Fb1_sig, Fb2_sig ]) # NON-POISSONIAN BACKGROUNDS

xsec_test_arr = np.logspace(-30, -15, 101)
#xsec_test_arr = [xsec_inj] #np.logspace(-23, -17, 10)
N = len(xsec_test_arr)
my_N = np.ones(comm.size) * int(N/comm.size)
my_N[:N%comm.size] += 1
my_N = (my_N).astype(np.int32)

my_xsec_test_arr = xsec_test_arr[np.sum(my_N[:comm.rank]):np.sum(my_N[:comm.rank+1])]

print(comm.rank, my_xsec_test_arr)
for xsec_t in my_xsec_test_arr:
    ll = 0
    for ib in range(len(my_iebins)-1):
        SCDb_arr = []
        if PPnoxsec_ebins[ib] != 0:
            ix = np.argmin(np.abs(SCD_params_xsec - xsec_t))

            ## FLOATING NORM, MID SLOPE

            Fb1 = (np.array(Fb_ebins_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])))[0] #*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]))[0]
            Fb2 = (np.array(Fb_ebins_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])))[1] #*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]))[1]
#            Fb3 = (np.array(Fb_ebins_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]))[2]

#            Fb1_lb = (np.array(Fb_ebins_lb_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]))[0]
#            Fb2_lb = (np.array(Fb_ebins_lb_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]))[1]

#            Fb1_ub = (np.array(Fb_ebins_ub_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]))[0]
#            Fb2_ub = (np.array(Fb_ebins_ub_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]))[1]

            scipy_min = minimize( lambda args: ll_func( xsec_t, ix, args[0], args[1], 0, -2-args[2], blazar_SCD[1], -3+args[3], blazar_SCD[3], blazar_SCD[4], blazar_SCD[5], A_ebins_xsec[ix][ib]/((xsec_t/SCD_params_xsec[ix])), Fb1, Fb2 ),
                                  [0.1, 0.1, -2-blazar_SCD[0], 3+blazar_SCD[2] ], bounds = [ [0,10], [0,10], [0,8], [0,6] ], method="L-BFGS-B", options={'maxiter':10000, 'ftol': 1e-10, 'eps':1e-5, 'disp':True} ) 
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            print(ll)
            print( ll_func( xsec_t, ix, 0., 0.14331159, 0, blazar_SCD[0], blazar_SCD[1], blazar_SCD[2], blazar_SCD[3], blazar_SCD[4], blazar_SCD[5], A_ebins_xsec[ix][ib]/((xsec_t/SCD_params_xsec[ix])), Fb1, Fb2 ) )
            SCDb_arr_ebins.append(SCDb_arr)
    ll_arr.append(ll)
    print( xsec_t, ll )
ll_arr = np.array(ll_arr)
np.save("MPITemp/ll_"+str(comm.rank)+"_"+tag, np.array(ll_arr))
np.save("MPITemp/SCDb_"+str(comm.rank)+"_"+tag, np.array(SCDb_arr_ebins))
comm.Barrier()

if comm.rank == 0:
    ll_arr = np.empty(len(xsec_test_arr))
    SCD_arr_ebins = []
    for i in range(comm.size):
        ll_arr[np.sum(my_N[:i]):np.sum(my_N[:i+1])] = np.load("MPITemp/ll_"+str(i)+"_"+tag+".npy")
        SCD_arr_ebins.append(np.load("MPITemp/SCDb_"+str(i)+"_"+tag+".npy"))
    TS_xsec_ary = 2*(ll_arr - ll_arr[0])
    max_loc = np.argmax(TS_xsec_ary)
    max_TS = TS_xsec_ary[max_loc]
    
    xsec_rec = 1e-50
    for xi in range(max_loc, len(xsec_test_arr)):
        val = TS_xsec_ary[xi] - max_TS
        if val < -2.71:
            scale = (TS_xsec_ary[xi-1]-max_TS+2.71)/(TS_xsec_ary[xi-1]-TS_xsec_ary[xi])
            xsec_rec = xsec_test_arr[xi-1] + scale*(xsec_test_arr[xi] - xsec_test_arr[xi-1])
            break
    np.savez("lim_"+str(xsec_inj) + "_" + tag, xsec_rec, ll_arr, SCDb_arr_ebins )
    print("Recovered: ", xsec_rec)
    print("Best fit: ", xsec_test_arr[max_loc])

    
