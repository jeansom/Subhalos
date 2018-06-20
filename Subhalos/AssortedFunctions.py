from scipy.stats import chi2
import numpy as np

def PErrors( data ):
    err_dn = np.nanmedian(data, axis=0) - chi2.ppf(.32/2, 2*np.nanmedian(data, axis=0))/2. 
    err_up = chi2.ppf(1-.32/2, 2*(1+np.nanmedian(data, axis=0)))/2. - np.nanmedian(data, axis=0)
    return err_dn, err_up
    
def PandGErrors(data, poisson=False):
    err_dn, err_up = PErrors(data)
    err_std_up = np.std(data, axis=0)
    err_std_dn = np.std(data, axis=0)
    for iF, Fv in enumerate(np.nanmedian(data, axis=0)):
        if Fv <= 1 or poisson: 
            err_std_up[iF] = err_up[iF]
            err_std_dn[iF] = err_dn[iF]
    return err_std_dn, err_std_up
    
def myLog(n):
    return np.where( n!=0, np.log10(n), 0);