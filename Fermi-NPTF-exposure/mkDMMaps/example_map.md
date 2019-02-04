

```python
%matplotlib inline
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sys, os

import matplotlib.pyplot as plt
import matplotlib as mpl

import triangle

from matplotlib import rc
import pymultinest, triangle

import healpy as hp
import pandas as pd


mpl.rcParams.update({'font.size': 18, 'font.family': 'serif'})


###Make sure you append the git directory
sys.path.append('/Users/bsafdi/Dropbox/Edep-NPTF/github/NPTF-ID-Catalog/mkDMMaps/')
###New modules to load in
import NFW
import mkDMMaps
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


# Load in Tully catalog (top 30)


```python
#pd.read_csv("2MRSTully_top50.csv")
```


```python
tully_top_30 = pd.read_csv("2MRSTully_top50.csv")
```

# Making a J-factor map for Tully (Example)

This code puts down halos on the sky, including their spatial extension.  It should be modified at some point to include the PSF while putting down the halos.


```python
z_array = tully_top_30.z.values #[0.01]
r_s_array = tully_top_30.rvir.values*1e-3 #Convert to Mpc#[1.0]
J_array = 10**tully_top_30.logJ.values #[1e9]
ell_array = tully_top_30.l.values*2*np.pi/360 #[0.0]
b_array = tully_top_30.b.values*2*np.pi/360 #[0.0]
nside=128 #256
angle_mult=2 #You will look at pixels within angle_mult*psi_0 radians of the source, where tan(psi_0) = r_s / d, and d is the distance to the source

final_map = np.zeros(hp.nside2npix(nside)) #This is the final J-factor map

for i in range(len(z_array)):
    #print mk.psi_0*mk.angle_mult #to see how far out we will go.  Other pixels are just fixed to zero
    #The following is the main class
    mk = mkDMMaps.mkDMMaps(z = z_array[i],r_s = r_s_array[i], J_0 = J_array[i],ell = ell_array[i],b = b_array[i],nside=nside)
    final_map += mk.map #Here, we add to our final map the specific map for the i^th halo.
```


```python
#hp.cartview(final_map,lonra=[-10,10],latra=[-10,10])
```

Now let's plot the map


```python
hp.mollview(final_map,max=1e16)
```


![png](output_9_0.png)



```python

```
