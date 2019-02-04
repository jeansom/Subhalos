from scipy.stats import chi2
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np
import healpy as hp

def PErrors( data ):
    err_dn = np.nanmedian(data, axis=0) - chi2.ppf(.32/2, 2*np.nanmedian(data, axis=0))/2. 
    err_up = chi2.ppf(1-.32/2, 2*(1+np.nanmedian(data, axis=0)))/2. - np.nanmedian(data, axis=0)
    return err_dn, err_up
    
def PandGErrors(data, poisson=False):
    err_dn, err_up = PErrors(data)
    err_std_up = np.std(data, axis=0)
    err_std_dn = np.std(data, axis=0)
    for iF, Fv in enumerate(np.nanmedian(data, axis=0)):
        if Fv <= 10 or poisson or False: 
            err_std_up[iF] = err_up[iF]
            err_std_dn[iF] = err_dn[iF]
    return err_std_dn, err_std_up
    
def myLog(n):
    return np.where( n!=0, np.log10(n), 0);

def getEnergyBinnedMaps(name, cur_dir, iebins, ave=False, int32=False, nside=128):
    arr_ebins = []
    for ib, b in enumerate(iebins[:-1]):
        arr = np.zeros(hp.nside2npix(nside))
        n = 0
        for bin_ind in range(b, iebins[ib+1]):
            n+=1
            arr += np.load(cur_dir+'/'+name+str(bin_ind)+'.npy')
        arr_ebins.append(arr)
        if ave: arr_ebins.append(arr/n)
        if int32: arr_ebins.append(arr.astype(np.int32))
    return arr_ebins

import pandas as pd
ebins = 2*np.logspace(-1,3,41)[0:41]

def getPPnoxsec(mass, channel, iebins, cur_dir):
    dNdLogx_df = pd.read_csv(cur_dir+'/fermi_data/AtProduction_gammas.dat', delim_whitespace=True)
    dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(mass)))))[['Log[10,x]',channel]]
    Egamma = np.array(mass*(10**dNdLogx_ann_df['Log[10,x]']))
    dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
    dNdE_interp = interp1d(Egamma, dNdEgamma)
    PPnoxsec_ebins = []
    for ib, b in enumerate(iebins[:-1]):
        ebins_temp = [ ebins[b], ebins[iebins[ib+1]] ]
        if ebins_temp[0] < mass:
            if ebins_temp[1] < mass:
                # Whole bin is inside
                PPnoxsec_ebins.append(1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], ebins_temp[1])[0])
            else:
                # Bin only partially contained
                PPnoxsec_ebins.append(1.0/(8*np.pi*mass**2)*quad(lambda x: dNdE_interp(x), ebins_temp[0], mass)[0])
        else: PPnoxsec_ebins.append(0)
    return PPnoxsec_ebins