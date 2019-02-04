
# coding: utf-8

# In[2]:

import healpy as hp
import mpmath as mp
import matplotlib.cm as cm

import logging
logger = logging.getLogger(__name__)


# In[3]:

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


# In[4]:

from scipy import integrate, interpolate

r_0 = 8.5
r_s             = 20.   #NFW scale radius in kpc
gamma_nfw       = 1.25  #gNFW power, use values from Daylan et al. fits
sigma_v         = 1.7   #in 10^-26 cm^3 / s
rho_0           = 0.3   #in GeV / cm^3
m_chi           = 35.   #in GeV
photons_per_ann = 1.67  #1.67 for 1.9-11.9 GeV, 3.63 for 1.-11.9 GeV
avg_exposure = 4.E10

conv_factor = sigma_v * rho_0**2. / (8. * np.pi * m_chi**2.) * photons_per_ann * avg_exposure * 3.08E-5
rho_0_factor = np.power(r_0/r_s, gamma_nfw) * np.power(1. + r_0/r_s, 3. - gamma_nfw)

def rho_dimless(r):
    #r in kpc
    return np.power(r/r_s, -gamma_nfw) * np.power(1. + r/r_s, gamma_nfw - 3.) * rho_0_factor     

def r_NFW(l, psi_deg):
    return np.sqrt(r_0**2. + l**2. - 2.*r_0*l*np.cos(np.radians(psi_deg)))
    
def L_NFW_integral(psi_deg):
    return integrate.quad(lambda l: rho_dimless(r_NFW(l, psi_deg))**2., 0., 100.*r_s)[0]
    
def make_NFW_intensity_map():
    psi_deg = np.arange(0., 180.5, 0.5)
    intensity_NFW = conv_factor * np.vectorize(L_NFW_integral)(psi_deg)
    flux_NFW = interpolate.interp1d(psi_deg, intensity_NFW * 4.*np.pi / NPIX)
    psi_deg_pixels = np.array([np.degrees(np.arccos(np.dot([1.0, 0.0, 0.0], hp.pix2vec(NSIDE, pix)))) for pix in range(NPIX)])
    return flux_NFW(psi_deg_pixels)


# In[5]:

def mask_lat_band(lat_deg_min, lat_deg_max, nside):
    #make mask of band lat_deg_min < b < lat_deg_max
    mask_none = np.arange(hp.nside2npix(nside))
    return (np.radians(lat_deg_min) <= hp.pix2ang(nside, mask_none)[0]) *             (hp.pix2ang(nside, mask_none)[0] <= np.radians(lat_deg_max))

def mask_not_lat_band(lat_deg_min, lat_deg_max, nside):
    #make mask of region outside band lat_deg_min < b < lat_deg_max
    return np.logical_not(mask_lat_band(lat_deg_min, lat_deg_max, nside))

def mask_ring(ring_deg_min, ring_deg_max, center_theta_deg, center_phi_deg, nside):
    #make mask of region outside ring_deg_min < theta < ring_deg_max
    mask_none = np.arange(hp.nside2npix(nside))
    return (np.cos(np.radians(ring_deg_min)) >= np.dot(hp.ang2vec(np.radians(center_theta_deg), np.radians(center_phi_deg)), hp.pix2vec(nside, mask_none))) *             (np.dot(hp.ang2vec(np.radians(center_theta_deg), np.radians(center_phi_deg)), hp.pix2vec(nside, mask_none)) >= np.cos(np.radians(ring_deg_max)))


# In[6]:

NSIDE = 128
NPIX = 12*NSIDE**2

k_max = 30

sigma_PSF_deg = 0.2
sigma_PSF = sigma_PSF_deg*np.pi/180.

deg_GC  = 1
deg_min = 2
deg_max = 10

gmt_um = hp.read_map('sim/128-galprop.fits', verbose=False)
gmt_um_mean = np.mean(gmt_um)
mask_5_10 = np.logical_not(mask_ring(5, 10, 90, 0, NSIDE))
mask_no_GC = mask_ring(0, deg_GC, 90, 0, NSIDE)
mask_for_all_maps = np.logical_not(mask_ring(deg_min, deg_max, 90, 0, NSIDE)) + mask_lat_band(87.5, 92.5, NSIDE)
gmt = hp.ma(gmt_um - gmt_um_mean)
gmt.mask = mask_for_all_maps
dmmc = gmt.compressed()
x_iso_true = np.mean(gmt_um)

npixROI = NPIX - np.sum(mask_for_all_maps)
npix_5_10 = NPIX - np.sum(mask_5_10)

NFW_um = make_NFW_intensity_map()
NFW = hp.ma(NFW_um, copy=True)
NFW.mask = mask_for_all_maps
NFW_5_10 = hp.ma(NFW_um, copy=True)
NFW_5_10.mask = mask_5_10
NFW_no_GC = hp.ma(NFW_um, copy=True)
NFW_no_GC.mask = mask_no_GC + mask_for_all_maps


# In[7]:

import itertools

def jitter_smooth(counts_map, jitter_c=False):
    smoothed_map = np.zeros(len(counts_map))
    non_empty_pixels = np.arange(len(counts_map))[counts_map > 0]
    counts_in_non_empty_pixels = counts_map[counts_map > 0]
    counts_split_indices = np.cumsum(counts_in_non_empty_pixels)
    n_non_empty_pixels = np.sum(counts_map > 0)
    n_counts = np.sum(counts_in_non_empty_pixels)
    dtheta_c = np.zeros(n_non_empty_pixels)
    dphi_c = np.zeros(n_non_empty_pixels)
    if jitter_c:
        dtheta_c = np.random.normal(scale=sigma_PSF/10., size=n_non_empty_pixels)
        dphi_c = np.random.normal(scale=sigma_PSF/10., size=n_non_empty_pixels)
    theta_c, phi_c = hp.pix2ang(NSIDE, non_empty_pixels)
    theta = theta_c + dtheta_c
    phi = phi_c + dphi_c
    
    dtheta_jitter = np.split(np.random.normal(scale=sigma_PSF, size=n_counts), counts_split_indices[:-1])
    dphi_jitter = np.split(np.random.normal(scale=sigma_PSF, size=n_counts), counts_split_indices[:-1])
    theta_jitter = np.array([np.mod(theta[pixel] + dtheta_jitter[pixel], np.pi)
                             for pixel in range(n_non_empty_pixels)])
    theta_jitter_flat = np.fromiter(itertools.chain.from_iterable(theta_jitter), np.float64)
    phi_jitter = np.array([np.mod(phi[pixel] + dtheta_jitter[pixel], 2*np.pi)
                           for pixel in range(n_non_empty_pixels)])
    phi_jitter_flat = np.fromiter(itertools.chain.from_iterable(phi_jitter), np.float64)
    pixels_jitter = hp.ang2pix(NSIDE, theta_jitter_flat, phi_jitter_flat)
    pj_lo, pj_hi = pixels_jitter.min(), pixels_jitter.max()
    pj_bins = np.bincount(pixels_jitter - pj_lo)
    smoothed_map[pj_lo:pj_hi+1] += pj_bins
    return smoothed_map

def draw_xyz(num_IGPS, delta_IGPS, r_min, r_max):
    A = (3. - delta_IGPS)/(np.power(r_max, 3. - delta_IGPS) - np.power(r_min, 3. - delta_IGPS))
    def r(Y):
        return np.power((3. - delta_IGPS)*Y/A + np.power(r_min, 3. - delta_IGPS), 1./(3. - delta_IGPS))
    Y = np.random.uniform(size=num_IGPS)
    xyz = np.random.normal(size=(num_IGPS, 3))
    xyz_unit = np.divide(xyz, np.linalg.norm(xyz, axis=1)[:, None])
    return np.multiply(r(Y)[:, None], xyz_unit)
    
def draw_lum(num_IGPS, alpha_lum, lum_min, lum_max):
    A = (1. - alpha_lum)/(np.power(lum_max, 1. - alpha_lum) - np.power(lum_min, 1. - alpha_lum))
    def lum(Y):
        return np.power((1. - alpha_lum)*Y/A + np.power(lum_min, 1. - alpha_lum), 1./(1. - alpha_lum))
    Y = np.random.uniform(size=num_IGPS)
    return lum(Y)


# In[8]:

delta_IGPS  = 2.5        #IG PS radial PDF n(r) ~ r^-delta_IGPS
r_max       = 3.5         #distance from GC to truncate IG PS
r_0         = 8.5        #distance to GC in kpc
alpha_lum   = 1.5        #dN/dL ~ L^-alpha_lum (~1.2 for pulsars, according to Cholis et al.)
lum_min     = 0.02      #minimum L cutoff, units of 10^36 photon/s (over 1.9-11.9 GeV), 0.002 ~ 10^31.5 erg/s G100
lum_max     = 5.0         #maximum L cutoff, units of 10^36 photon/s (over 1.9-11.9 GeV), 75 ~ 10^36 erg/s G100

nsims = 50.
avg_exposure = 4.E10

lum_mean = (alpha_lum - 1.) * (lum_max**(2.-alpha_lum) - lum_min**(2.-alpha_lum)) /             ((2. - alpha_lum) * (lum_min**(1.-alpha_lum) - lum_max**(1.-alpha_lum)))
   
def r_GC(l, psi):
    return np.sqrt(r_0**2. + l**2. - 2.*r_0*l*np.cos(psi))

kpcincm = 3.086E21
kpcincm2div36 = kpcincm**2/1E36
r_s = 20.

A_const = integrate.quad(lambda r: 4.*np.pi * r**(-delta_IGPS + 2.), 0., r_max)[0]
    
B_const = integrate.nquad(lambda psi, l: 2.*np.pi * r_GC(l, psi)**-delta_IGPS * np.sin(psi) * (r_GC(l, psi) <= r_max),
                          [[np.radians(5.), np.radians(10.)], [0., 4.*r_s]], 
                          opts=[{'epsrel':1.E-3},{'epsrel':1.E-3}])[0] \
                            / (4.*np.pi * kpcincm2div36)

print A_const, B_const

#number of inner-galaxy PS at < r_max
num_IGPS = np.int(A_const * np.mean(NFW_5_10) * npix_5_10 / (B_const * avg_exposure * lum_mean)) 
print 'Simulating {0:d} simulations with {1:d} sources within {2:.1f} kpc each...'.format(int(nsims), num_IGPS, r_max)


# In[9]:

xyz = draw_xyz(num_IGPS*int(nsims), delta_IGPS, 0., r_max)
#transform 3D positions to LOS distances
xyz_shifted = xyz + np.array([r_0, 0., 0.])
lLOS = np.linalg.norm(xyz_shifted, axis=1)
#draw from luminosity distribution (units 10^34 ph/s integrated over energy range)
lum = draw_lum(num_IGPS*int(nsims), alpha_lum, lum_min, lum_max)
#use LOS distances and luminosities distribution to get fluxes in ph/cm^2/s
flux = lum /(4. * np.pi * np.power(lLOS, 2.) * kpcincm2div36) 
#sum fluxes in pixels to get mean flux map
theta = np.arccos(xyz_shifted[:,2]/lLOS)
phi = np.arctan2(xyz_shifted[:,1], xyz_shifted[:,0])
pixel = hp.ang2pix(NSIDE, theta, phi)
IGPS_mean_flux_map = np.zeros(NPIX)
IGPS_tot_flux_maps = np.zeros((nsims, NPIX))
IGPS_maps = np.zeros((nsims, NPIX))
IGPS_maps_PSF = np.zeros((nsims, NPIX))
for sim in range(int(nsims)):
    for IGPS in range(num_IGPS):
        IGPS_tot_flux_maps[sim][pixel[sim*num_IGPS + IGPS]] += flux[sim*num_IGPS + IGPS]*avg_exposure

    IGPS_mean_flux_map += IGPS_tot_flux_maps[sim] / nsims

    IGPS_maps[sim] = np.array([np.random.poisson(flux_in_pixel) for flux_in_pixel in IGPS_tot_flux_maps[sim]])
    IGPS_maps_PSF[sim] = jitter_smooth(IGPS_maps[sim])*np.logical_not(mask_for_all_maps)
    if (sim + 1) % 25 == 0:
            print 'Asimov IGPS iteration ' + str(sim + 1) + ' done.'


# In[10]:

npix_map = hp.ma(np.arange(NPIX))
npix_map.mask = mask_for_all_maps
NFW_maps = np.zeros((int(nsims), NPIX))
NFW_maps_PSF = np.zeros((int(nsims), NPIX))

for sim in range(int(nsims)):
    temp_map = np.array([np.random.poisson(flux_in_pixel) for flux_in_pixel in NFW.compressed()])
    put(NFW_maps[sim], npix_map.compressed(), temp_map)
    NFW_maps_PSF[sim] = jitter_smooth(NFW_maps[sim])*np.logical_not(mask_for_all_maps)
    if (sim + 1) % 25 == 0:
        print 'Asimov NFW iteration ' + str(sim + 1) + ' done.'


# In[11]:

npix_map = hp.ma(np.arange(NPIX))
npix_map.mask = mask_for_all_maps
diff_maps = np.zeros((int(2*nsims), NPIX))
diff_maps_PSF = np.zeros((int(2*nsims), NPIX))

diff_rand = 0.01

for sim in range(int(2*nsims)):
    temp_map = np.array([np.random.poisson(flux_in_pixel + x_iso_true) for flux_in_pixel 
                         in np.random.uniform(low=1.-diff_rand,high=1.+diff_rand, size=len(dmmc))*dmmc])
    put(diff_maps[sim], npix_map.compressed(), temp_map)
    diff_maps_PSF[sim] = jitter_smooth(diff_maps[sim])*np.logical_not(mask_for_all_maps)
    if (sim + 1) % 25 == 0:
        print 'Asimov diffuse iteration ' + str(sim + 1) + ' done.'


# In[11]:

print np.sum(np.mean(IGPS_maps_PSF, axis=0)*np.logical_not(mask_5_10))         / np.sum(np.mean(NFW_maps_PSF, axis=0)*np.logical_not(mask_5_10))

print np.mean(np.mean(IGPS_maps_PSF, axis=0))         / np.mean(np.mean(NFW_maps_PSF, axis=0))


# In[12]:

skymap_clip = 15

hp.cartview(np.mean(IGPS_maps_PSF, axis=0), min=0, max=skymap_clip, lonra=[-10,10], latra=[-10,10], title='point sources only')
hp.cartview(np.mean(NFW_maps_PSF, axis=0), min=0, max=skymap_clip, lonra=[-10,10], latra=[-10,10], title='dark matter only')


# In[13]:

# rcParams.update({'font.size': 28})
skymap_clip = 15
max2 = 30

# hp.cartview(np.mean(IGPS_maps_PSF, axis=0), min=0, max=skymap_clip, lonra=[-10,10], latra=[-10,10], title='point sources only')
# hp.cartview(np.mean(NFW_maps_PSF, axis=0), min=0, max=skymap_clip, lonra=[-10,10], latra=[-10,10], title='dark matter only')

hp.cartview(IGPS_maps_PSF[0], min=0, max=skymap_clip, lonra=[-10,10], latra=[-10,10], title='point sources only')
hp.cartview(NFW_maps_PSF[0], min=0, max=skymap_clip, lonra=[-10,10], latra=[-10,10], title='dark matter only')

hp.cartview(diff_maps_PSF[0] + IGPS_maps_PSF[0], min=0, max=max2, lonra=[-10,10], latra=[-10,10], title='diffuse + point sources')
hp.cartview(diff_maps_PSF[0] + NFW_maps_PSF[0], min=0, max=max2, lonra=[-10,10], latra=[-10,10], title='diffuse + dark matter')


# In[14]:

all_maps = np.zeros((2*nsims, NPIX))
all_maps[:nsims] = IGPS_maps_PSF + diff_maps_PSF[:nsims]
all_maps[nsims:] = NFW_maps_PSF + diff_maps_PSF[nsims:]
all_data = np.sort(np.compress(np.logical_not(mask_for_all_maps), all_maps, axis=1), axis=1)#[:, :-100]


# In[39]:

ntest = 10
ntrain = nsims - ntest
# training_maps = np.zeros((2*ntrain, NPIX))
# training_maps[:ntrain] = IGPS_maps_PSF[:ntrain] + diff_maps_PSF[:ntrain]
# training_maps[ntrain:] = NFW_maps_PSF[:ntrain] + diff_maps_PSF[nsims:nsims + ntrain]
training_data = np.concatenate((all_data[:ntrain], all_data[nsims:nsims + ntrain]))
training_target = np.zeros(2*ntrain)
training_target[ntrain:] = 1
training_names = ['pulsars', 'dark matter']
pulsar_test_data = all_data[ntrain:nsims]
DM_test_data = all_data[nsims+ntrain:]
test_data = np.concatenate((pulsar_test_data, DM_test_data))
test_target = np.zeros(2*ntest)
test_target[ntest:] = 1


# In[42]:

for data in training_data[:ntrain]:
    plot(data, 'r')
for data in training_data[ntrain:]:
    plot(data, 'g')
yscale('log')    
show()


# In[44]:

plot(mean(training_data[:ntrain], axis=0), 'r')
plot(mean(training_data[ntrain:], axis=0), 'g')
xscale('log')
yscale('log')
show()


# In[20]:

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

# from sklearn.svm import LinearSVC
# clf = LinearSVC()


# In[17]:

clf.fit(training_data, training_target)


# In[107]:

print clf.predict(pulsar_test_data)
print clf.predict(DM_test_data)


# In[108]:

print np.mean(np.sum(all_data[:nsims], axis=1))/np.mean(np.sum(all_data[nsims:], axis=1))


# In[109]:

from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True).fit(training_data)


# In[110]:

X_pca = pca.transform(training_data)


# In[111]:

from itertools import cycle

def plot_PCA_2D(data, target, target_names):
    colors = cycle('rgbcmykw')
    target_ids = range(len(target_names))
    figure()
    for i, c, label in zip(target_ids, colors, target_names):
        scatter(data[target == i, 0], data[target == i, 1],
                c=c, label=label)
    legend()


# In[112]:

plot_PCA_2D(X_pca, training_target, training_names)


# In[113]:

from mpl_toolkits.mplot3d import Axes3D

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(training_data)
Y = training_target
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[114]:

plot_PCA_2D(pca.transform(test_data), test_target, training_names)


# In[ ]:



