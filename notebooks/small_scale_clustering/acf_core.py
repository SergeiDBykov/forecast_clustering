import sys
sys.path.append('../../')

import treecorr
from astropy.table import Table, Column
from astropy.io import fits
import healpy as hp
from scripts.utils import set_mpl,rep_path, path2plots
from tqdm import tqdm
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from copy import copy
from scripts.forecast import DataGenerator
from scripts.luminosity_functions import def_agn_xlf
import pyccl as ccl
fiducial_params = {'Omega_c': 0.25, 'Omega_b': 0.05,
                   'h': 0.7, 'sigma8': 0.8, 'n_s': 0.96,
                   'transfer_function': 'boltzmann_camb',
                   'baryons_power_spectrum': 'nobaryons',
                   'matter_power_spectrum': 'halofit'}

def_cosmo = ccl.Cosmology(**fiducial_params)
savepath = rep_path+'results/data/small_scale_clustering'

# a good example of treecorr on the artificial fields: https://github.com/anikhalder/Integrated_Bispectrum/blob/master/power_spectrum_gaussian/power_spectrum_gaussian_single_patch.ipynb
set_mpl()


def hp_plot(map: np.ndarray):
    ''' plot the healpix map'''
    fig = plt.figure(figsize=(20, 15))
    hp.mollview(map, xsize=4000, fig=1)
    hp.graticule()
    return fig


def radec2thetaphi(ra: np.ndarray, dec: np.ndarray):
    # https://emfollow.docs.ligo.org/userguide/tutorial/skymaps.html
    theta = np.deg2rad(90. - dec)
    phi = np.deg2rad(ra)
    return theta, phi


def thetaphi2radec(theta: np.ndarray, phi: np.ndarray):
    # https://emfollow.docs.ligo.org/userguide/tutorial/skymaps.html
    ra = np.rad2deg(phi)
    dec = 90. - np.rad2deg(theta)
    return ra, dec


def pixs2radec(pix_idxs: np.ndarray, nside: int=512):
    ''' convert healpix pixel index to ra, dec'''
    theta, phi = hp.pixelfunc.pix2ang(nside, pix_idxs)
    ra, dec = thetaphi2radec(theta, phi)
    return ra, dec


def generate_healpix_src_map(cell: np.ndarray, density_deg2: float, nside: int=512) -> np.ndarray:
    """
    generate_healpix_src_map Generate a healpix map of the source density

    Args:
        cell (np.ndarray): input angular power spectrum without noise
        density_deg2 (float): density of the sources in deg^-2
        nside (int, optional): nside of the final map. Defaults to 512.

    Returns:
        np.ndarray: generated map of source counts in pixel
    """
    overdensity = hp.synfast(cell, nside, lmax=2*nside,
                             pol=False)  # 40 sec for nside = 4096
    overdensity = np.array(overdensity)
    #overdensity_ln = np.exp(np.array(overdensity)) - 1
    #print('lognormal overdensity!!!')
    # number of objects per pixel
    lam = density_deg2 * hp.nside2pixarea(nside, degrees=True)
    src_map = lam*(overdensity+1.)

    n_neg = np.sum(src_map < 0)
    assert 100*n_neg / \
        len(
            src_map) < 10, f"over 10% percent of negative points: {100*n_neg/len(src_map)} per cent  of total {len(src_map)}"
    src_map[src_map < 0] = 0
    src_map_poiss = np.array(stats.poisson.rvs(src_map))  #sample Poisson distribution in all pixels type: ignore

    observed_density = np.sum(src_map_poiss)/(4*np.pi*np.rad2deg(1)**2)
    print(f"observed density: {observed_density}")
    print(f"expected density: {density_deg2}")

    for i in range(int(np.max(src_map_poiss))):
        print(f"{i} src per pixel: {np.sum(src_map_poiss == i)}, or {np.sum(src_map_poiss == i)*100/len(src_map_poiss):.3e} per cent")

    return src_map_poiss


def get_srcmap_ero(slim: float=5e-15,
                   zmin: float=1.0,
                   zmax: float=1.1,
                   sigma_0: float=0.03,
                   f_fail: float=0.1,
                   nside: int =512,)-> Tuple[np.ndarray, float, DataGenerator, np.ndarray]:
    """
    get_srcmap_ero for a given parameters of AGN population, generates a healpix map of the source density (i.e. number of sources in each pixel)

    Args:
        slim (float, optional): limiting flux of the survey. Defaults to 5e-15.
        zmin (float, optional): minimum redshift. Defaults to 1.0.
        zmax (float, optional): maximum redshift. Defaults to 1.1.
        sigma_0 (float, optional): sigma_0 of photo-z. Defaults to 0.03.
        f_fail (float, optional): f-fail of photo-z. Defaults to 0.1.
        nside (int, optional): nside of the final map. Defaults to 512.

    Returns:
        Tuple: angular power spectrum, density of the sources in deg^-2, DataGenerator, healpix map of the source density
    """

    powspec_pars_dict_agn = {
        'slim': slim,
        'sigma_0': sigma_0,
        'f_fail': f_fail,
        'l_min': 1,
        'l_max': nside*2,
        'log_bins': -1,
        'fsky': 1,
        'has_rsd': True,
        'use_weighed_bias': False,
        'density_multiplier': 1.3,
        'camb_llimber': 110,
        'xlf': def_agn_xlf,
        'use_camb': False,
        'delta_i': 0,
        'remove_ignored_cells': True,
    }

    bin_left_edges = [zmin, zmax]
    powspec_pars_dict_agn['bin_left_edges'] = bin_left_edges

    datagen = DataGenerator(
        fiducial_params=fiducial_params,)

    datagen.invoke(
        **powspec_pars_dict_agn, plot_cell=False, plot_dndz=False)

    datagen.plot_cls(iis=[0])
    datagen.plot_dndzs()

    cell = datagen.cls_rebin[0]
    dens = datagen.src_dens_list[0]

    # try:
    #    pois = hp.read_map(f'{savepath}/{dataname}.fits')
    # except FileNotFoundError:
    #hp.write_map(f'{savepath}/{dataname}.fits', pois)
    pois = generate_healpix_src_map(cell, dens, nside=nside)

    plt.figure()
    hp.mollview(pois, xsize=1000)
    hp.graticule()

    return cell, dens, datagen, pois


def make_mask_radec_cuts(nside: int, ramin: float, ramax: float, decmin: float, decmax: float):
    """
    Generates a square mask limited by RA/DEC.
    see  https://github.com/xuod/castor/blob/master/castor/cosmo.py
    Parameters
    ----------
    nside : int
        `nside` parameter for healpix.
    ramin, ramax : float
        Should be in degrees in [0., 360.]
    decmin, decmax: [type]
        Should be in degrees in [-90., +90.]
    Returns
    -------
    array
        Returns the square mask.
    """
    
    def radec2thetaphi(ra, dec):
        """
        Converts ra and dec (in degrees) to theta and phi (Healpix conventions, in radians)
        """
        theta = np.pi/2. - np.deg2rad(dec)
        phi = np.deg2rad(ra)
        return theta, phi

    # Make sure arguments are correct
    assert (0. <= ramin <= 360.) & (0. <= ramax <= 360.) & (
        ramin < ramax), f"0< {ramin} < {ramax} < 360"
    assert (-90. <= decmin <= 90.) & (-90. <=
                                      decmax <= 90.) & (decmin < decmax), f"-90<={decmin} < {decmax} <= 90"

    mask = np.zeros(hp.nside2npix(nside))
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    ths, phs = radec2thetaphi([ramin, ramax], [decmin, decmax])
    thmin, thmax = np.min(ths), np.max(ths)
    phmin, phmax = np.min(phs), np.max(phs)

    heal_indexes = np.where((th < thmax) & (th > thmin) & (
        ph > phmin) & (ph < phmax))[0]

    mask[heal_indexes] = 1.
    mask = mask.astype(bool)
    borders_dict = {'RA_right': ramax, 'RA_left': ramin,
                    'DEC_bottom': decmin, 'DEC_top': decmax}

    return heal_indexes, mask, borders_dict


def make_cat(poisson_map: np.ndarray, mask: np.ndarray, mask_idxs: np.ndarray) -> Table:
    nside = hp.npix2nside(len(poisson_map))
    poisson_map_mask = poisson_map[mask]
    ra, dec = pixs2radec(mask_idxs, nside=nside)
    sources = poisson_map_mask >= 1 #select only non-zero pixels. If two or more sources are in the same pixel, it appears as one source in the catalog.
    #n_interept = np.sum(poisson_map_mask >= 2)
    # print(
    #    f"Number of sources with overlap (pixel>=2) in the map: {n_interept} out of #{len(poisson_map_mask[poisson_map_mask>=1])}")
    cat = Table({'RA': ra[sources], 'DEC': dec[sources],
                 'Z': poisson_map_mask[sources]})
    return cat


def make_random_cat(borders_dict: dict, size: int=5000) -> Table:
    ''' makea a random catalog with `size` sources in the given borders `borders_dict`'''
    ra_left, ra_right, dec_bottom, dec_top = borders_dict.values()

    ra = np.random.uniform(ra_left, ra_right, size=size)

    costhetamin = np.cos(np.deg2rad(90 - dec_bottom))  # TODO check this!
    costhetamax = np.cos(np.deg2rad(90 - dec_top))

    dec = 90. - \
        np.rad2deg(np.arccos(np.random.uniform(
            costhetamin, costhetamax, size=size)))

    cat = Table({'RA': ra, 'DEC': dec, 'Z': np.ones_like(ra)})
    return cat


def tile_sky_map(nside: int, width: float=5, height: float=4, n_patches: int=100) -> Tuple:
    """
    tile_sky_map Generates a sky map with `n_patches` patches of size `width` x `height`

    Args:
        nside (int): nside of the map
        width (float, optional): width of the tile. Defaults to 5.
        height (float, optional): height of the tile. Defaults to 4.
        n_patches (int, optional): number of patches. Defaults to 100.

    Returns:
        Tuple: total mask, list of binary masks, list of dict of borders, list of pixel patches (indeces of pixels)
    """
    
    def radec2thetaphi(ra, dec):
        """
        Converts ra and dec (in degrees) to theta and phi (Healpix conventions, in radians)
        """
        theta = np.pi/2. - np.deg2rad(dec)
        phi = np.deg2rad(ra)
        return theta, phi

    mask = np.zeros(hp.nside2npix(nside))
    th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    mask_total = np.zeros(hp.nside2npix(nside), dtype=int)
    masks = []
    borders_dicts = []
    pixels_patches = []

    n_masks = 0
    i = 0
    ra0 = 1.
    dec0 = -25.
    for i in tqdm(np.arange(0, int(350//(width)), 2.5)):
        for j in np.arange(0, int(350//height), 2.5):

            ramin = ra0+i*width
            ramax = ra0+(i+1)*width
            decmin = dec0+j*height
            decmax = dec0+(j+1)*height

            # if ramax > ra0 or decmax > 90:
            #    continue
            if decmax > np.abs(dec0) or decmin < -np.abs(dec0):
                continue

            borders_dict = {'RA_right': ramax, 'RA_left': ramin,
                            'DEC_bottom': decmin, 'DEC_top': decmax}
            borders_dicts.append(borders_dict)

            assert (0. <= ramin <= 360.) & (0. <= ramax <= 360.) & (
                ramin < ramax), f"0< {ramin} < {ramax} < 360"
            assert (-90. <= decmin <= 90.) & (-90. <=
                                              decmax <= 90.) & (decmin < decmax), f"-90<={decmin} < {decmax} <= 90"

            mask = np.zeros(hp.nside2npix(nside))
            ths, phs = radec2thetaphi([ramin, ramax], [decmin, decmax])
            thmin, thmax = np.min(ths), np.max(ths)
            phmin, phmax = np.min(phs), np.max(phs)

            pixels_patch = np.where((th < thmax) & (th > thmin) & (
                ph > phmin) & (ph < phmax))[0] #pixels indexes in the tile

            mask[pixels_patch] = 1.
            mask = mask.astype(bool)

            mask_total[pixels_patch] = mask_total[pixels_patch] + 1.
            mask_i = np.zeros(hp.nside2npix(nside), dtype=bool)
            mask_i[pixels_patch] = True
            pixels_patches.append(pixels_patch)

            masks.append(mask_i)
            n_masks += 1
            if n_masks > n_patches:
                print(f"{n_masks} patches")
                return mask_total, masks, borders_dicts, pixels_patches

    #assert np.max(mask) == 1, f"max of mask is {np.max(mask)} (should be 1)"
    return mask_total, masks, borders_dicts, pixels_patches


def make_data_and_random_in_patch(
                                    poisson_map: np.ndarray,
                                    mask: np.ndarray, borders_dict: dict, pixels_patch: np.ndarray,
                                    plot: bool=True,
                                    N_rand: int=10,
                                    )-> Tuple:
    """
    make_data_and_random_in_patch make a catalog of sources and a catalog of random sources in a given patch

    Args:
        poisson_map (np.ndarray): map of sources per pixel
        mask (np.ndarray): mask array, a healpix map
        borders_dict (dict): dict of borders of the tile
        pixels_patch (np.ndarray): indeces of pixels of the tile
        plot (bool, optional): Whether to plot cat+random cat.  Defaults to True.
        N_rand (int, optional): random catalog would be N_rand times larger then the data catalog. Defaults to 10.

    Returns:
        catalog of sources, catalog of random sources, mask map 
    """

    ramin = borders_dict['RA_left']
    ramax = borders_dict['RA_right']
    decmin = borders_dict['DEC_bottom']
    decmax = borders_dict['DEC_top']


    cat = make_cat(poisson_map, mask, pixels_patch)
    cat_rand = make_random_cat(borders_dict, size=len(cat['Z'])*N_rand)

    if plot:
        plt.figure()
        #hp.mollview(poisson_map*mask, xsize=1000)
        dummy = poisson_map
        dummy = dummy + 0.0
        dummy[~mask] = hp.UNSEEN
        dummy = hp.ma(dummy)
        hp.gnomview(dummy, rot=((ramin + ramax)/2, (decmin +
                                                    decmax)/2,), reso=(ramax-ramin)/5, xsize=500, title='counts')
        hp.graticule(color='white')
        #plt.figure()
        #hp.mollview(mask, xsize=1000)
        #hp.graticule()

        plt.figure(figsize=(12, 12))
        plt.scatter(cat['RA'], cat['DEC'], c='g', s=30)
        plt.scatter(cat_rand['RA'], cat_rand['DEC'], c='gray', s=1, alpha=0.3)

    return cat, cat_rand, mask


def calc_cf(data: Table, rand: Table, 
            min_sep: float=5e-3, max_sep: float=4, 
            bin_size: float=0.5, plot: bool=False) -> Tuple:
    """
    calc_cf calculate the angular auto correlation function of a catalog given a random catalog

    Args:
        data (Table): data catalog
        rand (Table): random catalog
        min_sep (float, optional): min angular separation in deg. Defaults to 5e-3.
        max_sep (float, optional): max angular separation in deg. Defaults to 4.
        bin_size (float, optional): size of the bin. Defaults to 0.5.
        plot (bool, optional): whether to plot the results. Defaults to False.

    Returns:
        Tuple: ACF: r, xi, error, dd pairs
    """


    tree_data = treecorr.Catalog(
        ra=data['RA'], dec=data['DEC'], ra_units='deg', dec_units='deg', )

    tree_rand = treecorr.Catalog(
        ra=rand['RA'], dec=rand['DEC'], ra_units='deg', dec_units='deg', )

    dd = treecorr.NNCorrelation(
        min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units='degrees')
    dd.process(tree_data)

    dr = treecorr.NNCorrelation(
        min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units='degrees')
    dr.process(tree_data, tree_rand)

    rr = treecorr.NNCorrelation(
        min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units='degrees')
    rr.process(tree_rand)

    xi, varxi = dd.calculateXi(rr, dr)
    r = np.exp(dd.meanlogr)
    sig = np.sqrt(varxi)

    if plot:
        plt.figure(figsize=(12, 12))
        plt.plot(r, xi, color='blue')
        plt.errorbar(r, xi, yerr=sig,
                     color='blue', lw=0.5, ls='')
        plt.xscale('log')
        plt.yscale('symlog', linthresh=2e-3)

        plt.xlabel(r'$\theta$ (degrees)')
        plt.ylabel(r'$w(\theta)$')


    return r, xi, sig,  dd
