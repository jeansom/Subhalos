import numpy as np 
import units
    
def c200_SP( M200, r, r200 ): # Sanchez-Conde-Prada mass-concentration relation
    c_arr = [37.5153, -1.5093, 1.636 * 10**(-2), 3.66 * 10**(-4), -2.89237 * 10**(-5), 5.32 * 10**(-7)]
    c_arr.reverse()
    c200_val = np.polyval(c_arr, np.log(M200*units.h/units.M_s))
    return c200_val

def c200_S( M200, r, r200 ): # Distance dependent c-m relation
    alphaR = 0.286
    C1 = 119.75
    C2 = -85.16
    alpha1 = 0.012
    alpha2 = 0.0026
    return (r/(402 * units.kpc))**(-alphaR) * ( C1*(M200/units.M_s)**(-alpha1) + C2*(M200/units.M_s)**(-alpha2))
