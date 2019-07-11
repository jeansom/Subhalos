from __future__ import division
from __future__ import print_function


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
    np.array( [-5.4142492972786815, 2.172104345, 1.67422471, 1.219407335, 61.18005025, 0.05023496327014509] ),
    np.array( [-4.98619637477781, 2.24016924, 1.36771901, 0.598503459, 106.0815055, 0.06034676405662057] ),
    np.array( [-5.066038999192546, 2.3051384099999996, 1.30267301, 0.2820041945, 140.805936, 0.036773288527101156] ),
    np.array( [-4.76502241, 2.22269868, 1.14546229, 0.125770298, 124.07591, 0.0485669303582896] ),
    np.array( [-4.57571879, 2.05, 1.03360119, -0.374956899, 86.4358784, 0.058791639893678305] ),
    np.array( [-4.57134424, 2.05, 0.957247191, -0.752547573, 97.4831177, 0.056772856770029814] ),
    np.array( [-4.51326846, 2.05, 0.835773901, -1.03445536, 97.8904608, 0.06639793349076611] ),
    np.array( [-4.5594701, 2.05820366, 0.754631227, -1.68317832, 114.956968, 0.061816966778634366] ),
    np.array( [-4.54236738, 2.07019124, 0.632113389, -1.32596527, 125.187277, 0.0763350125833563] ),
    np.array( [-4.55208903, 2.07045348, 0.521428991, -1.39704795, 137.859196, 0.0867130876232369] ),
    np.array( [-4.59654015, 2.12875567, 0.425414999, -1.3690295, 159.795016, 0.09760536641151979] ),
    np.array( [-4.61134188, 2.13100741, 0.245861525, -1.38036074, 172.009221, 0.1259691251222031] ),
    np.array( [-4.59718001, 2.0990278, 0.0593279958, -1.33383551, 176.63565, 0.15757985226643614] ),
    np.array( [-4.58431185, 2.06655174, -0.549765447, -0.56790069, 170.16068346, 0.5496507278745021] ),
    np.array( [-4.63732532, 2.05031271, -0.857080848, -0.68655096, 184.244227, 0.6824977738444359] ),
    np.array( [-4.68435333, 2.06245759, -1.47686807, -0.89002684, 186.42380448, 0.7692676990394123] ),
    np.array( [-4.714422515649848, 2.05, -1.572507775, -1.443929775, 183.09088777, 0.7860410372327179] ),
    np.array( [-4.8692431, 2.05007773, -0.74783313, -1.62840241, 215.00913456, 0.8004900444783594] ),
    np.array( [-4.990738263317503, 2.0500000050000002, -0.61566255, -2.58494543, 217.56453428999998, 0.9123207858658938] ),
    np.array( [-5.14560829, 2.08266773, 0.229910002, -3.22390323, 247.07739611, 0.8971429262647495] ),
    np.array( [-5.30171833, 2.0566429, 0.61784026, -4.28141848, 253.63927277, 0.951195050703919] ),
    np.array( [-5.52372047, 2.17015404, -0.57010827, -6.02989334, 290.31426569, 0.9103590329323027] ),
    np.array( [-5.719179252641322, 2.0947778699999997, 0.54204283, -9.313187500000002, 279.28840021999997, 0.9628772937335278] ),
    np.array( [-5.9755476396314595, 2.0929481699999997, 0.312241114, -10.0, 290.06401884, 0.9844072891974414] ),
    np.array( [-6.28815968, 2.05901097, -0.237665732, -10.0, 289.02456454, 0.9891562595553492] ),
    np.array( [-6.6821251451281505, 2.05, -1.5401008900000002, -10.0, 275.254589945, 0.978528300904334] ),
    np.array( [-7.2309346, 2.05, -2.69479712, -10.0, 300.12239076, 0.9841772279369938] )
]

SCD_params_ub = [
    np.array( [-5.294422628198014, 2.405448324, 1.7694129304, 1.3072946723999999, 69.982199344, 1.0] ),
    np.array( [-4.755788921724302, 2.4996634572, 2.1180561232, 1.00268925, 134.84156332959998, 1.0] ),
    np.array( [-4.637615299279395, 2.4964123244, 1.3719782016000002, 0.5242493139200002, 176.13046348, 1.0] ),
    np.array( [-4.561485568660861, 2.5668654472, 1.2786781288, 0.8891663736, 167.71778500119999, 1.0] ),
    np.array( [-4.417559154409566, 2.1403070304, 1.1077260147999999, 0.02064301776399996, 107.62453532, 1.0] ),
    np.array( [-4.372503994213168, 2.1357544948, 1.0346483404, -0.28724810896, 119.57394699999998, 1.0] ),
    np.array( [-4.359599339830202, 2.1495173452, 0.9564603006000001, -0.4802483499600001, 128.66366, 1.0] ),
    np.array( [-4.425477156535638, 2.1349085028, 0.8498664711599999, -0.7650741718000003, 138.54469331999996, 1.0] ),
    np.array( [-4.392588471242543, 2.1650366572, 0.7232068170799999, 0.15502661479999996, 150.69334363999997, 1.0] ),
    np.array( [-4.412915921654218, 2.2166727203999996, 0.6417550533999999, 0.06001077334399971, 172.15379292, 1.0] ),
    np.array( [-4.471126966210068, 2.3193855244, 0.5776630347999999, 0.04426856118799993, 215.48443419999998, 1.0] ),
    np.array( [-4.453446791656222, 2.4106583855999997, 0.48487383104000004, -0.027370442040000022, 241.09105079999998, 1.0] ),
    np.array( [-4.470894708038034, 2.4517166088, 1.0488822199999985, -0.15533401840000002, 261.56206332, 1.0] ),
    np.array( [-4.420340990254046, 2.4424570159999996, 0.19943909119999986, -0.2948736844, 274.85551356, 1.0] ),
    np.array( [-4.508563469887177, 2.3930533363999995, 1.322105693199999, -0.23985535140000014, 269.07674716, 1.0] ),
    np.array( [-4.532014922216297, 2.4491981752, 1.211014864, -0.34347295600000005, 292.4790797024, 1.0] ),
    np.array( [-4.569238694882692, 2.3176861075999997, 2.5273401440000014, -0.6586374624, 269.77473277160004, 1.0] ),
    np.array( [-4.670981351305567, 2.3842114156, 2.6827400188, -0.95286705416, 294.70520337920004, 1.0] ),
    np.array( [-4.810043148283093, 2.3238771416, 2.7260256712000004, -1.2366585691999998, 296.5427662636, 1.0] ),
    np.array( [-4.9640922615079175, 2.2833267568, 2.7266135015999997, -1.9893179748, 300.31086289999996, 1.0] ),
    np.array( [-5.08420181240391, 2.2503651496, 2.9999960559999996, -3.0136073004000004, 299.132470884, 1.0] ),
    np.array( [-5.380585566599541, 2.7464158723999996, 2.9999814132, -3.223535768, 407.2461040047998, 1.0] ),
    np.array( [-5.562736273575674, 2.5004237472, 2.9999896776, -5.513274451999999, 369.8781613968001, 1.0] ),
    np.array( [-5.84730579529708, 2.8242884476000008, 2.9999999948, -6.517780719999999, 435.97198968000004, 1.0] ),
    np.array( [-6.0982572224248, 2.9344630903999995, 2.9884558732, -8.988520193200005, 486.4490146671998, 1.0] ),
    np.array( [-6.481995435507938, 2.4933292148000006, 2.8392899676000005, -9.999999945199999, 393.3455552084, 1.0] ),
    np.array( [-6.942423752411803, 3.5762376587999998, -1.4064816996000011, -10.0, 455.9017318523998, 1.0] )
]

SCD_params_lb = [
    np.array( [-5.539427727509249, 2.052323482, 1.5903412132, 1.0721929272, 52.5737825428, 0.029661101954491197] ),
    np.array( [-5.166118469909978, 2.108143914, 1.0633820996, 0.35225593951999995, 80.292035092, 0.04750549307708268] ),
    np.array( [-5.237669685080811, 2.0976757784, 1.1551791656, 0.082250913152, 75.3031864612, 0.05352556608207147] ),
    np.array( [-5.084237823873809, 2.05, 0.5936013847999999, -0.34960565892, 75.515977764, 0.05275955723399172] ),
    np.array( [-4.7247594265144075, 2.05, 0.94681628552, -0.9229692196799999, 68.004442944, 0.05530104672414043] ),
    np.array( [-4.711825656522021, 2.05, 0.8316021437600001, -1.4445289836000001, 71.26445924800001, 0.05957448817909621] ),
    np.array( [-4.681756826082913, 2.05, 0.7132945371199999, -2.4369043064, 80.40128025599999, 0.05824793400385655] ),
    np.array( [-4.677486181290163, 2.05, 0.63601281144, -3.6077638419999998, 95.5241224384, 0.055809290151365] ),
    np.array( [-4.645669705811776, 2.05, 0.32499304743999996, -4.4125453292, 103.89843175280001, 0.059682567390590385] ),
    np.array( [-4.680309278971466, 2.05, 0.021864033682399985, -3.5451742976, 111.88850500000001, 0.0758990371963035] ),
    np.array( [-4.763572181982164, 2.05, -0.5710496058400001, -4.6716752728, 132.30749696, 0.07454879930103535] ),
    np.array( [-4.788636200746409, 2.05, -1.2194513856, -7.2202308624, 134.49976616, 0.08024519156231244] ),
    np.array( [-4.803596325133172, 2.05, -0.8812294496000002, -8.7637196268, 136.8057757068, 0.08915793174815602] ),
    np.array( [-4.816575149450568, 2.05, -2.8089418268, -6.6358396028, 131.1829996412, 0.1325042666271679] ),
    np.array( [-4.8130901579754095, 2.05, -2.6475956515999997, -6.7651743652, 142.8750003328, 0.1080957022417964] ),
    np.array( [-4.883776311012653, 2.05, -2.76651929, -2.7491869576, 141.70872856600002, 0.15013914634359884] ),
    np.array( [-4.907483865750618, 2.05, -2.9834081392, -3.678617194, 136.398222488, 0.5952089067253817] ),
    np.array( [-4.995718843150419, 2.05, -2.8673712352000003, -4.2841019152, 143.7000908984, 0.9021742059221168] ),
    np.array( [-5.093992002922396, 2.05, -2.9273592716, -5.6175278696, 159.3677622368, 0.9717516465437295] ),
    np.array( [-5.214985538816698, 2.05, -2.5304418312, -7.31739222, 167.37188559560002, 0.928769557358538] ),
    np.array( [-5.369272747390553, 2.05, -2.7943105564, -9.99999998, 159.8857708256, 0.9720636533286499] ),
    np.array( [-5.597197162276739, 2.05, -2.8958674392, -10.0, 201.56826476320003, 0.7946853010736614] ),
    np.array( [-5.804060374838731, 2.05, -2.763004308, -10.0, 176.0318675516, 0.9683149889416185] ),
    np.array( [-6.1069586070056845, 2.05, -2.9692923363999997, -10.0, 176.1481325108, 0.9844893191870887] ),
    np.array( [-6.40044367241566, 2.05, -2.9644327512, -10.0, 165.5909429564, 0.9842688131977974] ),
    np.array( [-6.8627496869933395, 2.05, -2.9990648220000002, -10.0, 169.7067859516, 0.9555791474399264] ),
    np.array( [-7.562729522362034, 2.05, -2.9629537268, -10.0, 188.5941299296, 0.7386567148574636] )
]

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
    5.623413251903491e-18
])

blazar_SCD = np.array([-4.733327518453883, 2.4461263, 1.77293727, 1.48618555, 60.55017686, 0.1 ])
tag = args.tag 
mass = 100
mass_inj = 100
PPnoxsec0_ebins = np.array([0.0000238640102822424, 0.00000592280390900841])
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

xsec0 = 1e-22
subhalos = np.load('/tigress/somalwar/Subhaloes/Subhalos/MC/EinastoTemplate2.npy')
subhalos = subhalos/np.mean(subhalos)

subhalo_MC = []
if useSubhalo == None: 
    for ib, b in enumerate(my_iebins[:-1]):
        fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map_NFW_"+str(b)+"-"+str(my_iebins[ib+1])+"_"+str(SCD_params_xsec[np.argmin(np.abs(SCD_params_xsec - xsec_inj))])+".npy")*np.mean(exposure_ebins[ib])*xsec_inj/SCD_params_xsec[np.argmin(np.abs(SCD_params_xsec - xsec_inj))]
#        fake_data = np.load("/tigress/somalwar/Subhaloes/Subhalos/MC/subhalo_flux_map0_"+str(b)+"-"+str(my_iebins[ib+1])+"_.npy")*exposure_ebins[ib]*xsec_inj/xsec0
        fake_data = np.round(np.random.poisson(fake_data)).astype(np.int32)
        subhalo_MC.append(fake_data)
        subhalo_MC[-1][subhalo_MC[-1] > 1000] = 0
else: 
    for ib, b in enumerate(my_iebins[:-1]):
        subhalo_MC.append(np.load(useSubhalo+str(b)+"-"+str(my_iebins[ib+1])+".npy")*np.mean(exposure_ebins[ib])*xsec_inj/xsec0)
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
        #data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib]))
        data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib], bkg_min.x[0]*dif_ebins[ib], bkg_min.x[1]*iso_ebins[ib] ))
        #data_ebins.append(makeMockData( subhalo_MC[ib], blazars_ebins[ib], bkg_min.x[0]*dif_ebins[ib], bkg_min.x[1]*iso_ebins[ib], bkg_min.x[2]*psc_ebins[ib] ))
        #data_ebins.append(np.round(np.random.poisson(subhalo_MC[ib])).astype(np.int32))
#        data_ebins[-1][data_ebins[-1] > 1000] = 0
        np.save("MPITemp/fake_data"+str(ib)+"_"+tag, data_ebins[-1])
        #data_ebins[-1] = np.load("MPITemp/fake_data0_test.npy")
comm.Barrier()
if comm.rank != 0:
    for ib, b in enumerate(my_iebins[:-1]):
        data_ebins.append(np.load("MPITemp/fake_data"+str(ib)+"_"+tag+".npy"))

bkg_arr = []
for ib in range(len(my_iebins)-1):
#    bkg_arr.append([])
#    bkg_arr.append([ [ dif_ebins[ib], 'dif'], [iso_ebins[ib], 'iso'], [psc_ebins[ib], 'psc'] ] )
    bkg_arr.append([ [ dif_ebins[ib], 'dif'], [iso_ebins[ib], 'iso'] ])

bkg_arr_np = [[[np.ones(len(blazars_ebins[ib])), 'blaz']], [[np.ones(len(blazars_ebins[ib])), 'blaz']]]
#bkg_arr_np = [ [[]], [[]] ]

ll_ebins_xsec = []
A_ebins_xsec = []
Fb_ebins_xsec = []
n_ebins_xsec = []
for ix in range(len(SCD_params_arr)):
    ll_ebins, A_ebins, Fb_ebins, n_ebins = getNPTFitLL( data_ebins, exposure_ebins, mask, Nb, tag, bkg_arr, bkg_arr_np, subhalos, False, False, True, *SCD_params_arr[ix] )
    ll_ebins_xsec.append(ll_ebins)
    A_ebins_xsec.append(A_ebins)
    Fb_ebins_xsec.append(Fb_ebins)
    n_ebins_xsec.append(n_ebins)

ll_arr = []
SCDb_arr_ebins = []
def ll_func( xsec_t, ix, A_dif, A_iso, A_psc, Ab, n1b, n2b, n3b, Fb1b, Fb2b ): 
    #    return -ll_ebins[ib]([A_dif, Ab, n1b, n2b, n3b, Fb1b, Fb2b, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]) # ALL BACKGROUNDS
    return -ll_ebins_xsec[ix][ib]([A_dif, A_iso, Ab, n1b, n2b, n3b, Fb1b, Fb2b, A_ebins_xsec[ix][ib]/((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]) # NON-POISSONIAN BACKGROUNDS

#    return -ll_ebins[ib]([A_dif, A_iso, A_psc, Ab, n1b, n2b, n3b, Fb1b, Fb2b, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]) # ALL BACKGROUNDS

#    return -ll_ebins[ib]([A_dif, A_iso, A_psc, A_ebins[ib]/((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins[ib])*((xsec_t/xsec0)*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib])) ]) # POISSON BACKGROUNDS
#    return -ll_ebins_xsec[ix][ib]([A_ebins_xsec[ix][ib]/((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]), *(np.array(Fb_ebins_xsec[ix][ib])*((xsec_t/SCD_params_xsec[ix])*PPnoxsec_ebins[ib]/PPnoxsec0_ebins[ib]))]) # NO BACKGROUNDS

xsec_test_arr = np.logspace(-30, -15, 101)
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

            '''
            ## ALL BACKGROUNDS
            scipy_min = minimize( lambda args: ll_func( xsec_t, *args ),
                                  [0.1, 0.1, 0.1, -6, 5, 0, 0.1, 60, 0.5], method="SLSQP", bounds = [ [0, 20], [0, 20], [0, 20], [-10, -2], [2.05,10], [-3,3], [-10, 1.95], [1,100], [0.1, 1.] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            '''
            ## NON-POISSON BACKGROUNDS
            scipy_min = minimize( lambda args: ll_func( xsec_t, 0, 0, 0, *args ),
                                  [-6, 5, 0, 0.1, 60, 0.5], method="SLSQP", bounds = [ [-10, -2], [2.05,10], [-3,3], [-10, 1.95], [1,100], [0.1, 1.] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            print(scipy_min.x)
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            '''
            ## POISSON BACKGROUND
            scipy_min = minimize( lambda args: ll_func( xsec_t, *args, 0., 1, 1., 0.1, 1000, 0.5 ),
                                  [0.1, 0.1, 0.1], method="SLSQP", bounds = [ [0, 20], [0, 20], [0, 20] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            '''
            ## NO BACKGROUNDS
            ll += -ll_func( xsec_t, 0., 1, 1., 0.1, 1000000., 0.5 )
            SCDb_arr.append([0., 5, 0, 0.1, 60, 0.5])
            '''
            '''
            ## FIX BLAZARS

            ll += -ll_func( xsec_t, ix, 0, 0, 0, *blazar_SCD )
            SCDb_arr.append([ -4.77303859,  2.4897986 ,  1.80402724,  1.43714242, 65.10919031, 0.12527546 ])
            '''
            '''
            ## FLOATING NORM
            scipy_min = minimize( lambda args: ll_func( xsec_t, ix, 0, 0, 0, args[0], *blazar_SCD[1:]  ),
                                  [-6], method="SLSQP", bounds = [ [-10, -2] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            print(scipy_min.x)
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            '''
            ## FLOATING NORM, MID SLOPE
            scipy_min = minimize( lambda args: ll_func( xsec_t, ix, args[0], 0, 0, 0., blazar_SCD[1], blazar_SCD[2], blazar_SCD[3], blazar_SCD[4], blazar_SCD[5] ),
                                  [0.1], method="SLSQP", bounds = [ [0., 20.] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            print(scipy_min.x)
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            ## FLOATING NORM, MID SLOPE
            scipy_min = minimize( lambda args: ll_func( xsec_t, ix, args[0], args[1], 0, -2-args[2], blazar_SCD[1], -3+args[3], blazar_SCD[3], blazar_SCD[4], blazar_SCD[5] ),
                                  [1e-10, 1e-10, 1e-10, 1e-10], bounds = [ [0., 10.], [0.,10.], [0, 8], [0, 6] ], method="L-BFGS-B", options={'maxiter':10000, 'ftol': 1e-10, 'eps':1e-5, 'disp':True} ) 
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            ## FLOATING NORM, LOW SLOPE, MID SLOPE
            scipy_min = minimize( lambda args: ll_func( xsec_t, ix, 0, 0, 0, args[0], args[1], blazar_SCD[2], blazar_SCD[3], blazar_SCD[4], blazar_SCD[5] ),
                                  [-6, 10], method="SLSQP", bounds = [ [-10, -2], [2.05, 10] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            print(scipy_min.x)
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            '''
            ## FLOATING NORM, LOW SLOPE
            scipy_min = minimize( lambda args: ll_func( xsec_t, ix, 0, 0, 0, args[0], args[1], args[2], blazar_SCD[3], args[3], blazar_SCD[5]*blazar_SCD[4]/args[3] ),
                                  [-6, 10, 0., 180], method="SLSQP", bounds = [ [-10, -2], [2.05, 10], [-3., 3.], [1, 1000] ], options={'ftol':1e-15, 'eps':1e-5, 'maxiter':5000, 'disp':True} )
            print(scipy_min.x)
            ll += -scipy_min.fun
            SCDb_arr.append(np.array(scipy_min.x))
            '''
            print(ll)
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

    
