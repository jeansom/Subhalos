import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import global_var as gv
import masks
import hists

import logging
logger = logging.getLogger(__name__)

def simulate_diff():
    #simulate diffuse component from diffuse model map and save to file
    if gv.do_diff_sim:
        gv.map_diff = hp.ma([float(np.random.poisson(flux + gv.x_iso_true)) for flux in gv.gmt_um])
        gv.map_diff.mask = np.copy(gv.mask_for_all_maps)
        np.savetxt(gv.sim_dir + gv.file_tag + '-diff-iso-' + str(gv.x_iso_true) + '.dat', gv.map_diff, '%1.5f')
        logging.info('%s', 'Simulated diffuse component and saved to ' + gv.sim_dir + gv.file_tag + '-diff-iso-' + str(gv.x_iso_true) + '.dat')
    #load previous simulation of diffuse component from file
    else:
        gv.map_diff = hp.ma(np.loadtxt(gv.sim_dir + gv.file_tag + '-diff-iso-' + str(gv.x_iso_true) + '.dat'))
        gv.map_diff.mask = np.copy(gv.mask_for_all_maps)
        logging.info('%s', 'Loaded simulated diffuse component from ' + gv.sim_dir + gv.file_tag+ '-diff-iso-' + str(gv.x_iso_true) + '.dat') 

def mean_num_sources_in_pixel(A, n1, n2, Sb, Smin_Sb):
    #calculate mean number of sources per pixel from broken PL (analytic)
    return A * Sb * (1. / (n1 - 1.) + (Smin_Sb**(1. - n2) - 1.) / (n2 - 1.)) / gv.NPIX

def mean_flux_in_pixel(A, n1, n2, Sb):
    #calculate mean flux per pixel from broken PL (analytic)
    return A * (n2 - n1) * Sb**2 / ((n1 - 2.) * (n2 - 2) * gv.NPIX)

def cdf_brokenPL(S, A, n1, n2, Sb, Smin_Sb, mean):
    #calculate CDF of broken PL (analytic)
    if S < Sb:
        return (A * Smin_Sb * Sb / (gv.NPIX * mean)) * Smin_Sb**-n2 * (1. - (Smin_Sb * Sb / S)**(n2 - 1.)) / (n2 - 1.)
    return (A * Smin_Sb * Sb / (gv.NPIX * mean)) * (Smin_Sb**-n2 * (1. - Smin_Sb**(n2 - 1.)) / (n2 - 1) + \
            Smin_Sb**-1. * (1. - (Sb / S)**(n1 - 1.)) / (n1 - 1.))

def cdf_inv_brokenPL(c, A, n1, n2, Sb, Smin_Sb, mean, cdf_Sb):
    #calculate inverse CDF of broken PL (analytic)
    if c < cdf_Sb:
        return Smin_Sb * Sb * (1. - gv.NPIX * mean / (A * Smin_Sb * Sb) * Smin_Sb**n2 * c * (n2 - 1))**(1./(1. - n2))
    return Sb * ((n2 - n1 + (n1 - 1.) * Smin_Sb**(1. - n2)) / (n2 - 1.) - \
            gv.NPIX * mean / (A * Sb) * (n1 - 1.) * c)**(1. / (1. - n1))

def random_brokenPL(A, n1, n2, Sb, Smin_Sb, mean, cdf_Sb):
    #draw random variable from broken PL using inverse CDF method
    c = np.random.uniform()
    return cdf_inv_brokenPL(c, A, n1, n2, Sb, Smin_Sb, mean, cdf_Sb)
    
def random_counts_in_pixel(num_sources, A, n1, n2, Sb, Smin_Sb, mean, cdf_Sb):
    return np.sum([np.random.poisson(random_brokenPL(A, n1, n2, Sb, Smin_Sb, mean, cdf_Sb)) for source in range(num_sources)])

def simulate_PS():
    #simulate point-source component from broken PL and save to file
    PS_file = '-PS-' + str(gv.A_true) + '-' + str(gv.n1_true) + '-' + str(gv.n2_true) + '-' + str(gv.Sb_true) + '-' + str(gv.Smin_Sb) + '.dat'
    #cutoff Smin_Sb should be chosen to give a reasonable mean number of sources per pixel (~150 seems OK)
    if gv.do_PS_sim:
        #calculate mean number of sources per pixel (analytic)
        mean_true = mean_num_sources_in_pixel(gv.A_true, gv.n1_true, gv.n2_true, gv.Sb_true, gv.Smin_Sb)
        #calculate mean flux per pixel (analytic)
        mean_flux_true = mean_flux_in_pixel(gv.A_true, gv.n1_true, gv.n2_true, gv.Sb_true)
        #find CDF at break Sb (analytic -- need to break up CDF as piecewise function)
        cdf_Sb_true = cdf_brokenPL(gv.Sb_true, gv.A_true, gv.n1_true, gv.n2_true, gv.Sb_true, gv.Smin_Sb, mean_true)
        #generate random number of sources in each pixel
        map_num_PS = np.random.poisson(mean_true, gv.NPIX)
        logging.info('%s', 'Simulating point-source component with mean {0:.2f} sources and {1:.2f} photons per pixel...'.format(mean_true, mean_flux_true))
        #generate random fluxes in each pixel from generated number of sources in each pixel
        PS_fluxes = np.array([random_counts_in_pixel(num_in_pixel, gv.A_true, gv.n1_true, gv.n2_true, gv.Sb_true, gv.Smin_Sb, mean_true, cdf_Sb_true)
              for num_in_pixel in map_num_PS])
        #save to file
        np.savetxt(gv.sim_dir + gv.file_tag + PS_file, PS_fluxes, '%1.5f')
        #create masked map
        gv.map_PS = hp.ma(PS_fluxes)
        gv.map_PS.mask = np.copy(gv.mask_for_all_maps)
        logging.info('%s', 'Simulated point-source fluxes saved to ' + gv.sim_dir + gv.file_tag + PS_file)
    #load previous simulation of point-source component from file        
    else:
        gv.map_PS = hp.ma(np.loadtxt(gv.sim_dir + gv.file_tag + PS_file))
        gv.map_PS.mask = np.copy(gv.mask_for_all_maps)
        logging.info('%s', 'Loaded simulated point-source component from ' + gv.sim_dir + gv.file_tag + PS_file)

def draw_xyz(num_IGPS, delta_IGPS, r_max):
    A = (3. - delta_IGPS)/(np.power(r_max, 3. - delta_IGPS))
    def r(Y):
        return np.power((3. - delta_IGPS)*Y/A, 1./(3. - delta_IGPS))
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

def simulate_IGPS():
    #simulate inner-galaxy point-source component from dNdL and dNdr and save to file
    IGPS_file = '-IGPS.dat'
    if gv.do_IGPS_sim:
        #draw 3D positions from radial distribution
        xyz = draw_xyz(gv.num_IGPS, gv.delta_IGPS, gv.r_max)
        #transform 3D positions to LOS distances
        xyz_shifted = xyz + np.array([gv.r_0, 0., 0.])
        lLOS = np.linalg.norm(xyz_shifted, axis=1)
        #draw from luminosity distribution
        lum = draw_lum(gv.num_IGPS, gv.alpha_lum, gv.lum_min, gv.lum_max)
        #use LOS distances and luminosities distribution to get fluxes
        flux = lum /(4. * np.pi * np.power(lLOS, 2.))
        #sum fluxes in pixels to get mean flux map
        mean_flux_map = np.zeros(gv.NPIX)
        theta = np.arccos(xyz_shifted[:,2]/lLOS)
        phi = np.arctan2(xyz_shifted[:,1], xyz_shifted[:,0])
        pixel = hp.ang2pix(gv.NSIDE, theta, phi)
        for IGPS in range(gv.num_IGPS):
             mean_flux_map[pixel[IGPS]] += flux[IGPS]
        logging.info('%s', 'Simulating inner-galaxy point-source component...')
        #generate random fluxes in each pixel from generated number of sources in each pixel
        IGPS_fluxes = np.array([float(np.random.poisson(f)) for f in mean_flux_map])
        logging.info('%s', 'Simulated inner-galaxy point-source component with {0:.2f} total photons.'.format(np.sum(IGPS_fluxes)))
        #save to file
        np.savetxt(gv.sim_dir + gv.file_tag + IGPS_file, IGPS_fluxes, '%1.5f')
        #create masked map
        gv.map_IGPS = hp.ma(IGPS_fluxes)
        gv.map_IGPS.mask = np.copy(gv.mask_for_all_maps)
        logging.info('%s', 'Simulated inner-galaxy point-source fluxes saved to ' + gv.sim_dir + gv.file_tag + IGPS_file)
    #load previous simulation of inner-galaxy point-source component from file        
    else:
        gv.map_IGPS = hp.ma(np.loadtxt(gv.sim_dir + gv.file_tag + IGPS_file))
        gv.map_IGPS.mask = np.copy(gv.mask_for_all_maps)
        logging.info('%s', 'Loaded simulated inner-galaxy point-source component from ' + gv.sim_dir + gv.file_tag + IGPS_file)      

def simulate_all():
    #simulate or load components
    simulate_diff()
    simulate_PS()
    simulate_IGPS()
    #generate iso map with same mean as PS component ("fake iso")
#    mean_PS = np.mean(gv.map_PS)
    mean_PS = 10.
    gv.map_fakeiso = hp.ma(np.array([float(np.random.poisson(mean_PS)) for pix in gv.gmt_um]))
    gv.map_fakeiso.mask = np.copy(gv.mask_for_all_maps)
    gv.map_PS = hp.ma(gv.map_fakeiso)
    gv.map_PS.mask = np.copy(gv.mask_for_all_maps)
    #add simulated components to give total simulated map
    gv.map_all_um = np.array(gv.map_diff) + np.array(gv.map_PS) + np.array(gv.map_IGPS)
    gv.map_all = hp.ma(gv.map_all_um, copy=True)
    gv.map_all.mask = np.copy(gv.mask_for_all_maps)
    #smear total map
    gv.map_all_um_PSF = hp.smoothing(gv.map_all_um, sigma=gv.sigma_smoothing, verbose=False, regression=False)
    gv.map_all_PSF = hp.ma(gv.map_all_um_PSF, copy=True)
    gv.map_all_PSF.mask = np.copy(gv.mask_for_all_maps)
    #smear PS only map
    gv.map_PS_PSF = hp.smoothing(gv.map_PS, sigma=gv.sigma_smoothing, verbose=False, regression=False)
    gv.map_PS_PSF.mask = np.copy(gv.mask_for_all_maps)
    #smear IGPS only map
    gv.map_IGPS_PSF = hp.smoothing(gv.map_IGPS, sigma=gv.sigma_smoothing, verbose=False, regression=False)
    gv.map_IGPS_PSF.mask = np.copy(gv.mask_for_all_maps)
    
    #smear diff + iso map
    gv.map_diff_PSF = hp.smoothing(gv.map_diff, sigma=gv.sigma_smoothing, verbose=False, regression=False)
    gv.map_diff_PSF.mask = np.copy(gv.mask_for_all_maps)

    #smear fake iso map
    gv.map_fakeiso_PSF = hp.smoothing(gv.map_fakeiso, sigma=gv.sigma_smoothing, verbose=False, regression=False)
    gv.map_fakeiso_PSF.mask = np.copy(gv.mask_for_all_maps)
    logging.info('Simulated maps loaded.')
    
    logging.info('In the unmasked region containing {0:.2f} pixels, there are:'.format(gv.npixROI))
    logging.info('%s', 'Diffuse       SIM = {0:.2f} total photons and {1:.2f} photons per pixel'.format(np.sum(gv.map_diff.compressed()),
                                                                                                        np.sum(gv.map_diff.compressed())/gv.npixROI))
    logging.info('%s', 'Point sources SIM = {0:.2f} total photons and {1:.2f} photons per pixel'.format(np.sum(gv.map_PS.compressed()),
                                                                                                        np.sum(gv.map_PS.compressed())/gv.npixROI))
    logging.info('%s', 'IGPS          SIM = {0:.2f} total photons and {1:.2f} photons per pixel'.format(np.sum(gv.map_IGPS.compressed()),
                                                                                                        np.sum(gv.map_IGPS.compressed())/gv.npixROI))
                                                                                        
def plot_simulated_map_all():
    #plot histograms of simulated maps and save to file
    #simulated diff + iso + PS
    hists.plot_hist(gv.map_all.compressed(), loglog=True, title='Simulated data (PS): counts histogram', plot_file=gv.file_tag + '-sim-hist-PS.png')
    plt.clf()
    #simulated diff + iso + PS smeared with PSF
    hists.plot_hist(gv.map_all_PSF.compressed(), loglog=True, title='Simulated data (PS) + PSF: counts histogram', plot_file=gv.file_tag + '-sim-hist-PS-PSF.png')
    plt.clf()
    #simulated PS only
    hists.plot_hist(gv.map_PS.compressed(), loglog=True, title='Simulated PS only: counts histogram', plot_file=gv.file_tag + '-sim-hist-PS-only.png')
    plt.clf()
    #simulated PS only smeared with PSF
    hists.plot_hist(gv.map_PS_PSF.compressed(), loglog=True, title='Simulated PS only + PSF: counts histogram', plot_file=gv.file_tag + '-sim-hist-PS-only-PSF.png')
    plt.clf()
       
    #simulated diff + iso + fake iso
    hists.plot_hist(gv.map_diff.compressed() + gv.map_fakeiso.compressed(), loglog=True, title='Simulated data (fake isotropic): counts histogram', plot_file=gv.file_tag + '-sim-hist-fakeiso.png')
    plt.clf()
    #simulated diff + iso + fake iso smeared with PSF
    hists.plot_hist(gv.map_diff_PSF.compressed() + gv.map_fakeiso_PSF.compressed(), loglog=True, title='Simulated data (fake isotropic) + PSF: counts histogram', plot_file=gv.file_tag + '-sim-hist-fakeiso-PSF.png')
    plt.clf()
    #simulated fake iso only
    hists.plot_hist(gv.map_fakeiso.compressed(), loglog=True, title='Simulated fake isotropic only: counts histogram', plot_file=gv.file_tag + '-sim-hist-fakeiso-only.png')
    plt.clf()
    #simulated fake iso smeared with PSF
    hists.plot_hist(gv.map_fakeiso_PSF.compressed(), loglog=True, title='Simulated fake isotropic only + PSF: counts histogram', plot_file=gv.file_tag + '-sim-hist-fakeiso-only-PSF.png')
    plt.clf()
    
    #plot skymaps of simulated maps and save to file
    #simulated diff + iso + PS
    hp.mollview(gv.map_all, max=100, title='Simulated skymap: diffuse + isotropic + point sources (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map.png')
    #simulated diff + iso + PS smeared with PSF
    hp.mollview(gv.map_all_PSF, max=100, title='Simulated skymap + PSF: diffuse + isotropic + point sources (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-PSF.png')
    #simulated PS only
    hp.mollview(gv.map_PS, max=100, title='Simulated skymap: point sources (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-PS-only.png')
    #simulated PS only smeared with PSF
    hp.mollview(gv.map_PS_PSF, max=100, title='Simulated skymap + PSF: point sources (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-PS-only-PSF.png')
    
    #simulated diff + iso + IGPS
    hp.mollview(gv.map_all, max=100, title='Simulated skymap: diffuse + isotropic + point sources + IGPS (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-IGPS.png')
    #simulated diff + iso + IGPS smeared with PSF
    hp.mollview(gv.map_all_PSF, max=100, title='Simulated skymap + PSF: diffuse + isotropic + point sources + IGPS (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-IGPS-PSF.png')
    #simulated IGPS only
    hp.mollview(gv.map_IGPS, max=100, title='Simulated skymap: IGPS (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-IGPS-only.png')
    #simulated IGPS only smeared with PSF
    hp.mollview(gv.map_IGPS_PSF, max=100, title='Simulated skymap + PSF: IGPS (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-IGPS-only-PSF.png')
    
    #simulated diff + iso + fake iso
    hp.mollview(gv.map_diff + gv.map_fakeiso, max=100, title='Simulated skymap: diffuse + isotropic + fake isotropic (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-fakeiso.png')
    #simulated diff + iso + fake iso smeared with PSF
    hp.mollview(gv.map_diff_PSF + gv.map_fakeiso_PSF, max=100, title='Simulated skymap + PSF: diffuse + isotropic + fake isotropic (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-fakeiso-PSF.png')
    #simulated fake iso only
    hp.mollview(gv.map_fakeiso, max=100, title='Simulated skymap: fake isotropic (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-fakeiso-only.png')
    #simulated fake iso smeared with PSF
    hp.mollview(gv.map_fakeiso_PSF, max=100, title='Simulated skymap + PSF: fake isotropic (clipped at 100)')
    plt.savefig(gv.plots_dir + gv.file_tag + '-sim-map-fakeiso-only-PSF.png')
    logging.info('%s', 'Saved simulated histograms and skymaps to ' + gv.plots_dir + gv.file_tag)
    #plt.show()  
    plt.close()
