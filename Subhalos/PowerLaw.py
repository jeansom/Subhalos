import numpy as np

def simple_power(F, A, *args):
    results = []
    if (len(args) - 1) % 2 != 0: return np.nan
    N_b = int((len(args) - 1) / 2)
    n = -np.array(args[:N_b+1])
    Fb = np.array(args[N_b+1:])
    for i in reversed(range(len(Fb)-1)):
        Fb[i] = Fb[i+1] - np.abs(Fb[i])
    Fbv = np.array(Fb.copy())
    Fbv = np.concatenate(([Fbv[0]], Fbv))
    Fb = np.concatenate(([-np.inf], Fb, [np.inf]))


    if not hasattr(F, "__len__"): F = [F]
    for Fv in np.array(F):
        place = np.where(np.logical_and(Fv < Fb[1:], Fv >= Fb[:-1]) != 0)[0][0]
        if place == 0: results.append(A - n[0]*(Fv - Fbv[0]))
        elif place == 1: results.append(A - n[1]*(Fv - Fbv[1]))
        else: results.append(A - (n[place]*(Fv - Fbv[place]) - sum(n[i-1]*(Fbv[i]-Fbv[i-1]) for i in range(1, place+1))))
    return results

def loglike(x, y, f, *args, vmin=-np.inf, vmax=np.inf ):
    y = np.array(y)
    x = np.array(x)
    return -0.5*sum((np.logical_and(vmin <= np.array(x), np.array(x) <= vmax).astype(np.float64))*(y-f(x, *args))**2 for x, y in zip(x, y))

def genMinuitString( Nb, A_init, A_bounds, A_error, n_init, n_bounds, n_error, Fb_init, Fb_bounds, Fb_error, fix_nlast=None, fix_Fblast=None, vmin=-np.inf, vmax=np.inf):
    argument_list = "A"
    options = "A = "+A_init+", limit_A="+A_bounds+", error_A="+A_error
    for i in range(1, Nb+1): 
        argument_list = argument_list + ", n"+str(i)
        options += ", n"+str(i)+"="+n_init+", limit_n"+str(i)+"="+n_bounds+", error_n"+str(i)+"="+n_error
    argument_list += ", n"+str(Nb+1)
    if fix_nlast!=None: options += ", n"+str(Nb+1)+"="+fix_nlast+", fix_n"+str(Nb+1)+"=True"
    else: options += ", n"+str(Nb+1)+"="+n_init+", limit_n"+str(Nb+1)+"="+str(n_bounds)+", error_n"+str(Nb+1)+"="+n_error
    for i in range(1, Nb):
        argument_list += ", Fb"+str(i)
        options += ", Fb"+str(i)+"="+Fb_init+", limit_Fb"+str(i)+"="+Fb_bounds+", error_Fb"+str(i)+"="+Fb_error
    argument_list += ", Fb"+str(Nb)
    if fix_Fblast != None: options += ", Fb"+str(Nb)+"="+str(fix_Fblast)+", fix_Fb"+str(Nb)+"=True"
    else: options += ", Fb"+str(Nb)+"="+Fb_init+", limit_Fb"+str(i)+"="+Fb_bounds+", error_Fb"+str(i)+"="+Fb_error
    return ("minuit = Minuit(lambda %s: -PowerLaw.loglike(AssortedFunctions.myLog(F), AssortedFunctions.myLog(F**2*dN/dF), PowerLaw.simple_power, %s, %s, %s), %s)" % (argument_list, argument_list, "vmin="+str(vmin), "vmax="+str(vmax), options));