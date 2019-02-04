import numpy as np
import scipy as sp
import healpy as hp
import matplotlib.pyplot as plt

import global_var as gv
import masks
import hists

import logging
logger = logging.getLogger(__name__)

#===========================================================================  
# Diffuse
#===========================================================================

def simulate_diff():
    #simulate diffuse component from diffuse model map and save to file
    if gv.do_diff_sim:
        if gv.which_diff_model_sim == 0:
            logging.info('%s', 'Using Fermi diffuse model to simulate diffuse component.')
            gv.sim_map_diff = hp.ma([float(np.random.poisson(flux + gv.x_iso_true - gv.fmt_um_mean)) for flux in gv.fmt_um])
            np.savetxt(gv.sim_dir + 'fermi-diff-iso-' + str(gv.x_iso_true) + '.dat', gv.sim_map_diff, '%1.5f')
            logging.info('%s', 'Simulated diffuse component and saved to ' + gv.sim_dir + 'fermi-diff-iso-' + str(gv.x_iso_true) + '.dat')
        if gv.which_diff_model_sim == 1:
            logging.info('%s', 'Using Galprop diffuse model to simulate diffuse component.')
            gv.sim_map_diff = hp.ma([float(np.random.poisson(flux + gv.x_iso_true - gv.gmt_um_mean)) for flux in gv.gmt_um])
            np.savetxt(gv.sim_dir + 'galprop-diff-iso-' + str(gv.x_iso_true) + '.dat', gv.sim_map_diff, '%1.5f')
            logging.info('%s', 'Simulated diffuse component and saved to ' + gv.sim_dir + 'galprop-diff-iso-' + str(gv.x_iso_true) + '.dat')
        gv.sim_map_diff.mask = np.copy(gv.mask_for_all_maps)

    #load previous simulation of diffuse component from file
    else:
        if gv.which_diff_model_sim == 0:
            gv.sim_map_diff = hp.ma(np.loadtxt(gv.sim_dir + 'fermi-diff-iso-' + str(gv.x_iso_true) + '.dat'))
            logging.info('%s', 'Loaded simulated diffuse component from ' + gv.sim_dir + 'fermi-diff-iso-' + str(gv.x_iso_true) + '.dat')             
        if gv.which_diff_model_sim == 1:
            gv.sim_map_diff = hp.ma(np.loadtxt(gv.sim_dir + 'galprop-diff-iso-' + str(gv.x_iso_true) + '.dat'))
            logging.info('%s', 'Loaded simulated diffuse component from ' + gv.sim_dir + 'galprop-diff-iso-' + str(gv.x_iso_true) + '.dat') 
        gv.sim_map_diff.mask = np.copy(gv.mask_for_all_maps)
        
#===========================================================================  
# Isotropic PS  
#===========================================================================

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
    PS_file = 'PS-' + str(gv.A_true) + '-' + str(gv.n1_true) + '-' + str(gv.n2_true) + '-' + str(gv.Sb_true) + '-' + str(gv.Smin_Sb) + '.dat'
    #cutoff Smin_Sb should be chosen to give a reasonable mean number of sources per pixel (~150 seems OK)
    if gv.do_PS_sim:
        #calculate mean number of sources per pixel (analytic)
        mean_true = mean_num_sources_in_pixel(gv.A_true, gv.n1_true, gv.n2_true, gv.Sb_true, gv.Smin_Sb)
        #calculate mean flux per pixel (analytic)
        mean_flux_true = mean_flux_in_pixel(gv.A_true, gv.n1_true, gv.n2_true, gv.Sb_true)
        #find CDF at break Sb (analytic -- need to break up CDF as piecewise function)
        cdf_Sb_true = cdf_brokenPL(gv.Sb_true, gv.A_true, gv.n1_true, gv.n2_true, gv.Sb_true, gv.Smin_Sb, mean_true)
        #generate random number of sources in each pixel
        sim_map_num_PS = np.random.poisson(mean_true, gv.NPIX)
        logging.info('%s', 'Simulating point-source component with analytic mean {0:.2f} sources and {1:.2f} photons per pixel...'.format(mean_true, mean_flux_true))
        #generate random fluxes in each pixel from generated number of sources in each pixel
        PS_fluxes = np.array([float(random_counts_in_pixel(num_in_pixel, gv.A_true, gv.n1_true, gv.n2_true, gv.Sb_true, gv.Smin_Sb, mean_true, cdf_Sb_true))
              for num_in_pixel in sim_map_num_PS])
        logging.info('%s', 'Simulated point-source component has mean {0:.2f} sources and {1:.2f} photons per pixel.'.format(np.mean(sim_map_num_PS),
                                                                                                                               np.mean(PS_fluxes)))
        #save to file
        np.savetxt(gv.sim_dir + PS_file, PS_fluxes, '%1.5f')
        #create masked map
        gv.sim_map_PS = hp.ma(PS_fluxes)
        logging.info('%s', 'Simulated point-source fluxes saved to ' + gv.sim_dir + PS_file)
    #load previous simulation of point-source component from file        
    else:
        gv.sim_map_PS = hp.ma(np.loadtxt(gv.sim_dir + PS_file))
        logging.info('%s', 'Loaded simulated point-source component from ' + gv.sim_dir + PS_file)

#===========================================================================  
# IGPS     
#===========================================================================

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
    IGPS_file = 'IGPS-' + str(gv.num_IGPS) + '-' + str(gv.delta_IGPS) + '-' + str(gv.r_max) + '-' + str(gv.alpha_lum) + '-' + str(gv.lum_min) + '-' + str(gv.lum_max) + '.dat'
    if gv.do_IGPS_sim:
        logging.info('%s', 'Simulating inner-galaxy point-source component...')
        #draw 3D positions from radial distribution
        xyz = draw_xyz(gv.num_IGPS, gv.delta_IGPS, gv.r_max)
        #transform 3D positions to LOS distances
        xyz_shifted = xyz + np.array([gv.r_0, 0., 0.])
        lLOS = np.linalg.norm(xyz_shifted, axis=1)
        #draw from luminosity distribution (units 10^34 ph/s integrated over energy range)
        lum = draw_lum(gv.num_IGPS, gv.alpha_lum, gv.lum_min, gv.lum_max)
        #use LOS distances and luminosities distribution to get fluxes in ph/cm^2/s
        flux = lum /(4. * np.pi * np.power(lLOS, 2.) * 9.52E6)
        logging.info('%s', 'Maximum IGPS flux from a single source is {0:.2E} photons/cm^2/s.'.format(np.max(flux)))
        #sum fluxes in pixels to get mean flux map
        mean_flux_map = np.zeros(gv.NPIX)
        theta = np.arccos(xyz_shifted[:,2]/lLOS)
        phi = np.arctan2(xyz_shifted[:,1], xyz_shifted[:,0])
        pixel = hp.ang2pix(gv.NSIDE, theta, phi)
        for IGPS in range(gv.num_IGPS):
             mean_flux_map[pixel[IGPS]] += flux[IGPS]
        #generate random fluxes in each pixel from generated number of sources in each pixel -- assume average exposure over energy bins
        IGPS_fluxes = np.array([float(np.random.poisson(f*gv.avg_exposure)) for f in mean_flux_map])
        logging.info('%s', 'Simulated inner-galaxy point-source component with {0:.2f} total photons.'.format(np.sum(IGPS_fluxes)))
        #save to file
        np.savetxt(gv.sim_dir + IGPS_file, IGPS_fluxes, '%1.5f')
        #create masked map
        gv.sim_map_IGPS = hp.ma(IGPS_fluxes)
        logging.info('%s', 'Simulated inner-galaxy point-source fluxes saved to ' + gv.sim_dir + IGPS_file)
    #load previous simulation of inner-galaxy point-source component from file        
    else:
        gv.sim_map_IGPS = hp.ma(np.loadtxt(gv.sim_dir + IGPS_file))
        logging.info('%s', 'Loaded simulated inner-galaxy point-source component from ' + gv.sim_dir + IGPS_file)
        
#===========================================================================  
# NFW      
#===========================================================================
        
conv_factor = gv.sigma_v * gv.rho_0**2. / (8. * np.pi * gv.m_chi**2.) * gv.photons_per_ann * gv.avg_exposure * 3.08E-5
rho_0_factor = np.power(gv.r_0/gv.r_s, gv.gamma_nfw) * np.power(1. + gv.r_0/gv.r_s, 3. - gv.gamma_nfw)

def rho_dimless(r):
    #r in kpc
    return np.power(r/gv.r_s, -gv.gamma_nfw) * np.power(1. + r/gv.r_s, gv.gamma_nfw - 3.) * rho_0_factor     

def r_NFW(l, psi_deg):
    return np.sqrt(gv.r_0**2. + l**2. - 2.*gv.r_0*l*np.cos(np.radians(psi_deg)))
    
def L_NFW_integral(psi_deg):
    return sp.integrate.quad(lambda l: rho_dimless(r_NFW(l, psi_deg))**2., 0., 100.*gv.r_s)[0]
    
def flux_NFW(psi_deg):
    return conv_factor * 4.*np.pi * L_NFW_integral(psi_deg)/ gv.NPIX
    
def make_NFW_intensity_map():
    psi_deg = np.arange(0., 180.5, 0.5)
    intensity_NFW = conv_factor * np.vectorize(L_NFW_integral)(psi_deg)
    flux_NFW = sp.interpolate.interp1d(psi_deg, intensity_NFW * 4.*np.pi / gv.NPIX)
    psi_deg_pixels = np.array([np.degrees(np.arccos(np.dot([1.0, 0.0, 0.0], hp.pix2vec(gv.NSIDE, pix)))) for pix in range(gv.NPIX)])
    return flux_NFW(psi_deg_pixels)

def simulate_NFW():
    #simulate NFW component
    if gv.do_NFW_sim:
        logging.info('%s', 'Simulating NFW component...')
        gv.NFW_um = make_NFW_intensity_map()
        gv.sim_map_NFW = hp.ma([float(np.random.poisson(flux)) for flux in gv.NFW_um])
        logging.info('%s', 'Simulated NFW component with {0:.2f} total photons.'.format(np.sum(gv.sim_map_NFW)))
        np.savetxt(gv.sim_dir + 'NFW.dat', gv.sim_map_NFW, '%1.5f')
        logging.info('%s', 'Simulated NFW fluxes saved to NFW.dat')
    #load previous simulation of NFW component from file
    else:
        gv.sim_map_NFW = hp.ma(np.loadtxt(gv.sim_dir + 'NFW.dat'))
        logging.info('%s', 'Loaded simulated NFW component from ' + gv.sim_dir + 'NFW.dat')             
    
def simulate_all():
    #simulate or load components
    if gv.include_diff:
        simulate_diff()
    if gv.include_PS:
        simulate_PS()
    if gv.include_IGPS:    
        simulate_IGPS() 
    if gv.include_NFW:
        simulate_NFW()      
    gv.sim_map_PS.mask = np.copy(gv.mask_for_all_maps)    
    gv.sim_map_IGPS.mask = np.copy(gv.mask_for_all_maps)
    gv.sim_map_NFW.mask = np.copy(gv.mask_for_all_maps)       
    #add simulated components to give total simulated map
    gv.sim_map_all_um = np.array(gv.sim_map_diff) + np.array(gv.sim_map_PS) + np.array(gv.sim_map_IGPS) + np.array(gv.sim_map_NFW)
#    gv.sim_map_all_um = np.array(gv.sim_map_diff) + np.random.poisson(1., size=gv.NPIX) + np.array(gv.sim_map_IGPS) #use this to test for spurious PS recovery
    gv.sim_map_all = hp.ma(gv.sim_map_all_um, copy=True)
    gv.sim_map_all.mask = np.copy(gv.mask_for_all_maps)
    #smear total map
    gv.sim_map_all_um_PSF = hp.smoothing(gv.sim_map_all_um, sigma=gv.sigma_smoothing, verbose=False, regression=False)
    gv.sim_map_all_PSF = hp.ma(gv.sim_map_all_um_PSF, copy=True)
    gv.sim_map_all_PSF.mask = np.copy(gv.mask_for_all_maps)
    #smear diff + iso map
    gv.sim_map_diff_PSF = hp.smoothing(gv.sim_map_diff, sigma=gv.sigma_smoothing, verbose=False, regression=False)
    gv.sim_map_diff_PSF.mask = np.copy(gv.mask_for_all_maps)
    #smear PS only map
    gv.sim_map_PS_PSF = hp.smoothing(gv.sim_map_PS, sigma=gv.sigma_smoothing, verbose=False, regression=False)
    gv.sim_map_PS_PSF.mask = np.copy(gv.mask_for_all_maps)    
    #smear IGPS only map
    gv.sim_map_IGPS_PSF = hp.smoothing(gv.sim_map_IGPS, sigma=gv.sigma_smoothing, verbose=False, regression=False)
    gv.sim_map_IGPS_PSF.mask = np.copy(gv.mask_for_all_maps)
    #smear NFW only map
    gv.sim_map_NFW_PSF = hp.smoothing(gv.sim_map_NFW, sigma=gv.sigma_smoothing, verbose=False, regression=False)
    gv.sim_map_NFW_PSF.mask = np.copy(gv.mask_for_all_maps)

    logging.info('Simulated maps loaded.')
      
    logging.info('In the unmasked region containing {0:.2f} pixels, there are:'.format(gv.npixROI))
    logging.info('%s', 'Diffuse + Iso SIM = {0:.2f} total photons and {1:.2f} photons per pixel'.format(np.sum(gv.sim_map_diff.compressed()),
                                                                                                        np.sum(gv.sim_map_diff.compressed())/float(gv.npixROI)))
    logging.info('%s', 'Isotropic PS  SIM = {0:.2f} total photons and {1:.2f} photons per pixel'.format(np.sum(gv.sim_map_PS.compressed()),
                                                                                                        np.sum(gv.sim_map_PS.compressed())/float(gv.npixROI)))
    logging.info('%s', 'IG PS         SIM = {0:.2f} total photons and {1:.2f} photons per pixel'.format(np.sum(gv.sim_map_IGPS.compressed()),
                                                                                                        np.sum(gv.sim_map_IGPS.compressed())/float(gv.npixROI)))
    logging.info('%s', 'NFW           SIM = {0:.2f} total photons and {1:.2f} photons per pixel'.format(np.sum(gv.sim_map_NFW.compressed()),
                                                                                                        np.sum(gv.sim_map_NFW.compressed())/float(gv.npixROI)))                                                                                                        

    diff_sim_flux = np.mean(gv.sim_map_diff.compressed())
    PS_sim_flux = np.mean(gv.sim_map_PS.compressed())
    IGPS_sim_flux = np.mean(gv.sim_map_IGPS.compressed())
    all_sim_flux = np.mean(gv.sim_map_all.compressed())
    NFW_sim_flux = np.mean(gv.sim_map_NFW.compressed())    

    logging.info('%s', 'Diffuse + Iso SIM = {0:.2f}%'.format(diff_sim_flux/all_sim_flux*100))
    logging.info('%s', 'Isotropic PS  SIM = {0:.2f}%'.format(PS_sim_flux/all_sim_flux*100))
    logging.info('%s', 'IG PS         SIM = {0:.2f}%'.format(IGPS_sim_flux/all_sim_flux*100))
    logging.info('%s', 'All PS        SIM = {0:.2f}%'.format((PS_sim_flux + IGPS_sim_flux)/all_sim_flux*100))
    logging.info('%s', 'NFW           SIM = {0:.2f}%'.format(NFW_sim_flux/all_sim_flux*100))
                                                                                                                                                                                            
def plot_simulated_map_all():
#    #plot histograms of simulated maps and save to file
#    #simulated diff + iso + PS + IGPS
#    hists.plot_hist(gv.sim_map_all.compressed(), loglog=True, title='Simulated counts: diffuse + isotropic + all PS', plot_file='sim_plots/' + 'hist-all.png')
#    plt.clf()
#    #simulated diff + iso + PS + IGPS smeared with PSF
#    hists.plot_hist(gv.sim_map_all_PSF.compressed(), loglog=True, title='Simulated counts + PSF: diffuse + isotropic + all PS', plot_file='sim_plots/' + 'PSF-hist-all.png')
#    plt.clf()
#    #simulated diff + iso
#    hists.plot_hist(gv.sim_map_diff.compressed(), loglog=True, title='Simulated counts: diffuse + isotropic', plot_file='sim_plots/' + 'hist-diff-iso.png')
#    plt.clf()
#    #simulated diff + iso smeared with PSF
#    hists.plot_hist(gv.sim_map_diff_PSF.compressed(), loglog=True, title='Simulated counts + PSF: diffuse + isotropic', plot_file='sim_plots/' + 'PSF-hist-diff-iso.png')
#    plt.clf()
#    #simulated PS + IGPS only
#    hists.plot_hist(gv.sim_map_PS.compressed() + gv.sim_map_IGPS.compressed(), loglog=True, title='Simulated counts: isotropic PS + IG PS', plot_file='sim_plots/' + 'hist-PS-IGPS.png')
#    plt.clf()
#    #simulated PS only smeared with PSF
#    hists.plot_hist(gv.sim_map_PS_PSF.compressed() + gv.sim_map_IGPS_PSF.compressed(), loglog=True, title='Simulated counts + PSF: isotropic PS + IG PS', plot_file='sim_plots/' + 'PSF-hist-PS-IGPS.png')
#    plt.clf()

    #plot skymaps of simulated maps and save to file
    #simulated diff + iso + PS + IGPS
    hp.mollview(gv.sim_map_all_um, max=gv.skymap_clip, title='Simulated skymap: all components')
    plt.savefig(gv.sim_plots_dir + 'skymap-all-um.png')
    #simulated diff + iso + PS + IGPS smeared with PSF
    hp.mollview(gv.sim_map_all_um_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: all components')
    plt.savefig(gv.sim_plots_dir + 'PSF-skymap-all-um.png')

        
    if gv.skymap_plot_mode == 0:   
        #plot skymaps of simulated maps and save to file
        #simulated diff + iso + PS + IGPS
        hp.mollview(gv.sim_map_all, max=gv.skymap_clip, title='Simulated skymap: all components')
        plt.savefig(gv.sim_plots_dir + 'map-all.png')
        #simulated diff + iso + PS + IGPS smeared with PSF
        hp.mollview(gv.sim_map_all_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: all components')
        plt.savefig(gv.sim_plots_dir + 'PSF-map-all.png')

        #simulated diff + iso
        hp.mollview(gv.sim_map_diff, max=gv.skymap_clip, title='Simulated skymap: diffuse + isotropic')
        plt.savefig(gv.sim_plots_dir + 'map-diff-iso.png')
        #simulated diff + iso smeared with PSF
        hp.mollview(gv.sim_map_diff_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: diffuse + isotropic')
        plt.savefig(gv.sim_plots_dir + 'PSF-map-diff-iso.png')

        if gv.include_PS:
            #simulated PS only
            hp.mollview(gv.sim_map_PS, max=gv.skymap_clip, title='Simulated skymap: isotropic PS')
            plt.savefig(gv.sim_plots_dir + 'map-PS.png')
            #simulated PS + IGPS only smeared with PSF
            hp.mollview(gv.sim_map_PS_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: isotropic PS')
            plt.savefig(gv.sim_plots_dir + 'PSF-map-PS.png')
        
        if gv.include_IGPS:
        #simulated IGPS only
            hp.mollview(gv.sim_map_IGPS, max=gv.skymap_clip, title='Simulated skymap: IG PS')
            plt.savefig(gv.sim_plots_dir + 'map-IGPS.png')
            #simulated IGPS only smeared with PSF
            hp.mollview(gv.sim_map_IGPS_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: IG PS')
            plt.savefig(gv.sim_plots_dir + 'PSF-map-IGPS.png')
        
        if gv.include_PS and gv.include_IGPS:
            #simulated PS + IGPS only
            hp.mollview(gv.sim_map_PS + gv.sim_map_IGPS, max=gv.skymap_clip, title='Simulated skymap: isotropic PS + IG PS')
            plt.savefig(gv.sim_plots_dir + 'map-PS-IGPS.png')
            #simulated PS + IGPS only smeared with PSF
            hp.mollview(gv.sim_map_PS_PSF + gv.sim_map_IGPS_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: isotropic PS + IG PS')
            plt.savefig(gv.sim_plots_dir + 'PSF-map-PS-IGPS.png')
        
        if gv.include_NFW:
            #simulated NFW only
            hp.mollview(gv.sim_map_NFW, max=gv.skymap_clip, title='Simulated skymap: NFW')
            plt.savefig(gv.sim_plots_dir + 'map-NFW.png')
            #simulated IGPS only smeared with PSF
            hp.mollview(gv.sim_map_NFW_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: NFW')
            plt.savefig(gv.sim_plots_dir + 'PSF-map-NFW.png')
        
    if gv.skymap_plot_mode == 1:   
        #plot skymaps of simulated maps and save to file
        #simulated diff + iso + PS + IGPS
        hp.cartview(gv.sim_map_all, max=gv.skymap_clip, title='Simulated skymap: all components', lonra=[-20,20], latra=[-20,20])
        plt.savefig(gv.sim_plots_dir + 'map-all.png')
        #simulated diff + iso + PS + IGPS smeared with PSF
        hp.cartview(gv.sim_map_all_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: all components', lonra=[-20,20], latra=[-20,20])
        plt.savefig(gv.sim_plots_dir + 'PSF-map-all.png')

        #simulated diff + iso
        hp.cartview(gv.sim_map_diff, max=gv.skymap_clip, title='Simulated skymap: diffuse + isotropic', lonra=[-20,20], latra=[-20,20])
        plt.savefig(gv.sim_plots_dir + 'map-diff-iso.png')
        #simulated diff + iso smeared with PSF
        hp.cartview(gv.sim_map_diff_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: diffuse + isotropic', lonra=[-20,20], latra=[-20,20])
        plt.savefig(gv.sim_plots_dir + 'PSF-map-diff-iso.png')

        if gv.include_PS:
            #simulated PS only
            hp.cartview(gv.sim_map_PS, max=gv.skymap_clip, title='Simulated skymap: isotropic PS', lonra=[-20,20], latra=[-20,20])
            plt.savefig(gv.sim_plots_dir + 'map-PS.png')
            #simulated PS + IGPS only smeared with PSF
            hp.cartview(gv.sim_map_PS_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: isotropic PS', lonra=[-20,20], latra=[-20,20])
            plt.savefig(gv.sim_plots_dir + 'PSF-map-PS.png')
        
        if gv.include_IGPS:
            #simulated IGPS only
            hp.cartview(gv.sim_map_IGPS, max=gv.skymap_clip, title='Simulated skymap: IG PS', lonra=[-20,20], latra=[-20,20])
            plt.savefig(gv.sim_plots_dir + 'map-IGPS.png')
            #simulated IGPS only smeared with PSF
            hp.cartview(gv.sim_map_IGPS_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: IG PS', lonra=[-20,20], latra=[-20,20])
            plt.savefig(gv.sim_plots_dir + 'PSF-map-IGPS.png')
         
        if gv.include_PS and gv.include_IGPS:   
            #simulated PS + IGPS only
            hp.cartview(gv.sim_map_PS + gv.sim_map_IGPS, max=gv.skymap_clip, title='Simulated skymap: isotropic PS + IG PS', lonra=[-20,20], latra=[-20,20])
            plt.savefig(gv.sim_plots_dir + 'map-PS-IGPS.png')
            #simulated PS + IGPS only smeared with PSF
            hp.cartview(gv.sim_map_PS_PSF + gv.sim_map_IGPS_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: isotropic PS + IG PS', lonra=[-20,20], latra=[-20,20])
            plt.savefig(gv.sim_plots_dir + 'PSF-map-PS-IGPS.png')
            
        if gv.include_NFW:
            #simulated NFW only
            hp.cartview(gv.sim_map_NFW, max=gv.skymap_clip, title='Simulated skymap: NFW', lonra=[-20,20], latra=[-20,20])
            plt.savefig(gv.sim_plots_dir + 'map-NFW.png')
            #simulated NFW only smeared with PSF
            hp.cartview(gv.sim_map_NFW_PSF, max=gv.skymap_clip, title='Simulated skymap + PSF: NFW', lonra=[-20,20], latra=[-20,20])
            plt.savefig(gv.sim_plots_dir + 'PSF-map-NFW.png')
    
    plt.close()
