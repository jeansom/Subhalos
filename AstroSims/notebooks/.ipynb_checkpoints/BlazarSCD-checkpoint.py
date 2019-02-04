
# coding: utf-8

# In[1]:


# Load modules and stuff

import sys, os
sys.path.append("../code/")

from constants import *
import sources as sc

import numpy as np
from tqdm import *
from joblib import Parallel, delayed  
import multiprocessing
from ipywidgets import IntProgress

# In[2]:


# Ignore integration warnings

import warnings 
warnings.filterwarnings("ignore") 


# # Blazars

# In[3]:


# Initialize luminosity function class for blazars (here, BL Lacs and FSRQs based on 1501.05301)
# Simplifying assumptions here: Single power law energy spectrum, no EBL attenuation

LF_BL = sc.LuminosityFunctionBL(model='blazars', ebl=False, sed='pl', lcut=False)


# ## Spectrum

# In[4]:


# Get intensity spectrum

E_vals = np.logspace(np.log10(0.1),np.log10(100),48)*GeV # Energies to scan over

# E_vals = [int(E) for E in np.linspace(1,100,99)*GeV] # Energies to scan over

def return_BL_spec(E): # Just a wrapper for multi-core calculation
    return LF_BL.dIdE(E)

num_cores = multiprocessing.cpu_count() 
print "Calculating on", str(num_cores), "cores..."

dIdE_BL_vals = Parallel(n_jobs=num_cores)(delayed(return_BL_spec)(E) for E in E_vals)  

# Set spectrum values for interpolation
# LF_BL.set_dIdE(E_vals, dIdE_BL_vals)


# In[5]:


# Plot spectrum 


# ## Source count

# In[ ]:


# Define energy bins

CTB_en_bins = 10**np.linspace(np.log10(0.2), np.log10(2000),41)
CTB_bin_centers = [10**((np.log10(CTB_en_bins[i])+np.log10(CTB_en_bins[i+1]))/2) for i in range(len(CTB_en_bins)-1)]
CTB_bin_widths = [CTB_en_bins[i+1]-CTB_en_bins[i] for i in range(len(CTB_en_bins)-1)]


# In[14]:


# List of bins to get source counts for

bins = [10,15,20]


# In[ ]:


# Get theoretical source counts in the different bins defined above

dNdF_BL_vals = [[] for i in range(len(bins)-1)]

F_vals_bl = np.logspace(-14,-6,1000)*Centimeter**-2*Sec**-1 # Flux values to scan over

def return_BL_dNdF(F, E1, E2): # Just a wrapper for multi-core calculation
    return LF_BL.dNdFp(F,E1,E2)

num_cores = multiprocessing.cpu_count() 
print "Calculating on", str(num_cores), "cores..."

for i in (range(len(bins)-1)):
    print "Now doing bins", bins[i], "to", bins[i+1]
    dNdF_BL_vals[i] = Parallel(n_jobs=num_cores)(delayed(return_BL_dNdF)(F,CTB_en_bins[bins[i]]*GeV,CTB_en_bins[bins[i+1]]*GeV) for F in F_vals_bl) 


# In[ ]:


# In[ ]:


for i in range(len(bins)-1):
    np.save("/tigress/somalwar/Subhaloes/Subhalos/blazars/blazar_dNdF_"+str(bins[i])+"-"+str(bins[i+1]), [F_vals_bl/(Centimeter**-2*Sec**-1), np.array(dNdF_BL_vals[i])*(Centimeter**-2*Sec**-1)/srdeg2])

