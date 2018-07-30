import numpy as np
import units

# General cosmological constants for Milky Way
G = 43007.1 # Gravitational constant in units [(km/s)**2*(kpc/(1e10*M_s))]
H0 = .07 # Hubble in units of [(km/s)/(kpc/h)]
h = 0.7 # Dimensionless Hubble parameter
r0 = 8 # Position of sun in [kpc/h]
r_vir = 213.5 # r200 for MW in [kpc/h]

# Parameters for MW NFW profile
rho_c = 3*H0**2/(8*np.pi*G) # Critical density with units from above
delta_c = 200. # Virial overdensity