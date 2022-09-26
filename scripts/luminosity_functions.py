from typing import Callable, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import optimize
from scipy.interpolate import interp2d
import pyccl as ccl
from astropy import units as u
from astropy.units import Mpc, km, s, erg, cm, deg  # type: ignore
from astropy import constants as c
from astropy.constants import c
from .utils import rep_path
from tqdm import tqdm
from warnings import warn

sr2degsq = (180/np.pi)**2
cgs_flux = erg/cm**2/s  # type: ignore
cgs_lumin = erg/s  # type: ignore


# default cosmology
def_cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96,
                          transfer_function='boltzmann_camb', matter_power_spectrum='linear', baryons_power_spectrum='nobaryons')


def differential_comoving_volume(cosmo: ccl.Cosmology, z: np.ndarray) -> np.ndarray:
    """
    differential_comoving_volume calculates the differential comoving volume at given redshifts for the flat universe.
    dV/dz = c r(z)^2 / H(z) in Mpc^3/deg^2

    Args:
        cosmo (ccl.Cosmology): ccl cosmology object
        z (np.ndarray): array of redshifts

    Returns:
        np.ndarray: diffreential comoving volume at given redshifts, in Mpc^3/deg^2
    """
    if not hasattr(z, 'shape'):
        z = np.array([z])
    z = np.asarray(z)
    a = 1 / (1 + z)
    ccl_diff = c*(ccl.background.comoving_radial_distance(cosmo, a)*Mpc)**2 / (
        ccl.background.h_over_h0(cosmo, a)*cosmo['h']*100*(km/s)/Mpc)/u.sr
    ccl_diff = ccl_diff.to(Mpc**3/deg**2)
    return ccl_diff




def calc_po_flux(elo: float, ehi: float, gamma: float) -> float:
    """
    calc_po_flux: integrates a powerlaw spectrum (photon index gamma) with unit normalization
    between elo and ehi (keV),

    F=int from elo to ehi E*I(E)dE, where I(E)=E^(-gamma)

    used fo k-correction, see https://cxc.harvard.edu/sherpa4.6/threads/calc_kcorr/
    and for translation of XLF from one band to another


    Args:
        elo (float): lower energy
        ehi (float): upper energy
        gamma (float): photon index

    Returns:
        float: flux if a power law spectrum with no absorption
    """

    if gamma == 2:
        return np.log(ehi/elo)
    else:
        return (ehi**(2-gamma)-elo**(2-gamma))/(2-gamma)


def LBand1_to_LBand2_conv(band1: list, band2: list, gamma: float) -> float:
    """
    LBand1_to_LBand2_conv calculates the ratio of fluxes in band 1 to Band 2 .
    Therefore the luminoity in  band 1 is luminosity in  band 2 times that function.
    For instance, if gamma=1.9, L_0.5-2=0.74*L_2-10

    I checked with fig. 9 from Fotopoulou16 and see that in order to
    transform XLF originaly defined for BAND1 to that in BAND2, I use formula:
    f=LBand1_to_LBand2_conv(band1, band2,gamma)
    L_band1 = L_band2 * f
    therefore:

    XLF(L_band1, origina band of xlf)=XLF(L_band2, my band of interest * f)

    Args:
        band1 (list): first band in form [e_min, e_max]
        band2 (list): second band in the same form
        gamma (float): photon index

    Returns:
        float: conversion factor: ratio of two fluxes
    """

    return calc_po_flux(band1[0], band1[1], gamma)/calc_po_flux(band2[0], band2[1], gamma)


def L210_to_L052_conv(gamma: float) -> float:
    '''
    This means that the function calculates the ratio of fluxes in band 0.5-2 keV to 2-10 keV.
    Therefore the luminoity in 0.5-2 keV band is luminosity in 2-10 keV band times that function.
    For instance, if gamma=1.9, L_0.5-2=0.74*L_2-10
    '''
    return calc_po_flux(0.5, 2., gamma)/calc_po_flux(2., 10., gamma)


def k_correction_po(z: np.ndarray, gamma: float = 1.9) -> np.ndarray:
    """
    k_correction_po k-correction for a powerlaw does not depend on energies, only on redshift and gamma

    usage:
    https://dro.dur.ac.uk/11371/1/11371.pdf
    L_x=4pi D_L^2 F_X/(1+z)^(2-gamma),
    or
    L_x=4pi D_L^2 F_X / k_correction_po(z,gamma),

    L_rest frame = L_observed / K(z), where L_observed = 4pi d_L(z) Slim, see Kolodzig 2017 formula (10)

    Args:
        z (np.ndarray): array of Z where k-korrection of powerlaw spectrum flux is needed
        gamma (float, optional): photon index. Defaults to 1.9.

    Returns:
        np.ndarray: k correction at given z
    """

    return (z+1)**(2-gamma)


# the next two functions are logNlogS from the paper of Georgakakis 2008 for hard and soft bands, these are all objects, but dominated by AGN especially in hard band
def G08_best_fit_hard(f: np.ndarray,
                      f_ref: float = 1e-14,
                      beta1: float = -1.56, beta2: float = -2.52,
                      f_b: float = 10**(+0.09)*1e-14, K: float = 3.79e16) -> np.ndarray:
    """
    G08_best_fit_hard: best fit logNlogS from the paper of Georgakakis 2008 for hard 2-10 keV band. Those source counts are dominated by AGN

    Args:
        f (np.ndarray): flux
        f_ref (float, optional): parameter of logNlogS fitting. Defaults to 1e-14.
        beta1 (float, optional): parameter of logNlogS fitting. Defaults to -1.56.
        beta2 (float, optional): parameter of logNlogS fitting. Defaults to -2.52.
        f_b (float, optional): parameter of logNlogS fitting. Defaults to 10**(+0.09)*1e-14.
        K (float, optional): parameter of logNlogS fitting. Defaults to 3.79e16.

    Returns:
        np.ndarray: logNlogS at given flux
    """
    Kprime = K*(f_b/f_ref)**(beta1-beta2)

    return np.piecewise(f, [f < f_b, f >= f_b],
                        [lambda f: K*(f_ref/(1+beta1))*((f_b/f_ref)**(1+beta1) - (f/f_ref)**(1+beta1))-Kprime*f_ref/(1+beta2)*(f_b/f_ref)**(1+beta2), lambda f: -Kprime*f_ref/(1+beta2)*(f/f_ref)**(1+beta2)])


def G08_best_fit_soft(f: np.ndarray,
                      f_ref: float = 1e-14,
                      beta1: float = -1.58, beta2: float = -2.5,
                      f_b: float = 10**(-0.04)*1e-14, K: float = 1.51e16) -> np.ndarray:
    """
    G08_best_fit_soft: best fit logNlogS from the paper of Georgakakis 2008 for soft 0.5-2 keV band. Those source counts are dominated by AGN

    Args:
        f (np.ndarray): flux
        f_ref (float, optional): parameter of logNlogS fitting.
        beta1 (float, optional): parameter of logNlogS fitting.
        beta2 (float, optional): parameter of logNlogS fitting.
        f_b (float, optional): parameter of logNlogS fitting.
        K (float, optional): parameter of logNlogS fitting.

    Returns:
        np.ndarray: logNlogS at given flux
    """

    Kprime = K*(f_b/f_ref)**(beta1-beta2)

    return np.piecewise(f, [f < f_b, f >= f_b],
                        [lambda f: K*(f_ref/(1+beta1))*((f_b/f_ref)**(1+beta1) - (f/f_ref)**(1+beta1))-Kprime*f_ref/(1+beta2)*(f_b/f_ref)**(1+beta2), lambda f: -Kprime*f_ref/(1+beta2)*(f/f_ref)**(1+beta2)])


def Double_Power_Law(L: np.ndarray,
                     K0: float=6.69e-7,
                     L_star: float=10**(43.94),
                     gamma1: float=0.87, gamma2: float=2.57) -> np.ndarray:
    """
    Double_Power_Law 
     local AGN sample (z ~ 0)  follows a power law distribution with a turn over at low luminosities (Maccacaro et al., 1983, 1984). Default parameters are from Hasinger 2005, table 5, or Alex Kolodzig PhD, table 2.3 for LDDE model. Used as a auxillary function of other XLFs


    Args:
        L (np.ndarray): luminosity
        K0 (float, optional): norm constant. Defaults to 6.69e-7.
        L_star (float, optional): turnover L. Defaults to 10**(43.94).
        gamma1 (float, optional): po1. Defaults to 0.87.
        gamma2 (float, optional): po2. Defaults to 2.57.

    Returns:
        Callable: np.ndarray

        returns phi=dPhi(L,z=0)/dlogL., where phi is the 'ordinary' XLF, and Phi is defined below:
        Phi(L,z)=d^2N/dLdV_c is the number of objects (N) of given populsation per univ comoving volume V_c per unit Luminosity L,

        dV_c=c/H0 * D_M^2/E(z) dOmegadz, - differential comoving volume, defined by cosmology used
        D_M=c/H_0 int_0^z dx/E(x),
        E(x)=sqrt(Omega_matter(1+z)^3+Omega_energy)

        phi is in units Mpc^(-3)/dex (dex means the difference in powers of 10, for instance 10^44-10^43  is one dex)


        see also  Hogg2000, Fotopoulou Phd (1.15), Kolodzig Phd.

        The same definitions goes for all XLF defined below

    """



    xlf = K0/((L/L_star)**(gamma1) + (L/L_star)**(gamma2))
    return xlf


def ldde(L: np.ndarray, z: np.ndarray,
         K0: float=6.690e-7,
         L_star: float=10**(43.94),
         gamma1: float=0.87, gamma2: float=2.57,
         p1_44: float=4.7, p2_44: float=-1.5,
         beta1: float=0.7, beta2: float=0.6,
         z_cutoff_0: float=1.96331,
         L_alpha: float=10.**(44.67), alpha: float=0.21):
    '''
    This is Luminosity dependent density evolution as defined in H05 and a few other papers.
    All parameters are floats and from H05 (Hasiner 2005).
    L - luminosity
    z - redshift
    '''

    def p1(L):
        return p1_44+beta1*(np.log10(L)-44.)

    def p2(L):
        return p2_44+beta2*(np.log10(L)-44.)

    def z_cutoff(L):
        return np.piecewise(L, [L <= L_alpha, L > L_alpha], [lambda L: z_cutoff_0*(L/L_alpha)**(alpha), lambda L: z_cutoff_0])

    def evol_factor(L, z):
        zc = z_cutoff(L)

        return np.select([z <= zc, z > zc], [(1+z)**p1(L), (1+zc)**(p1(L))*((1+z)/(1+zc))**(p2(L))])

    return evol_factor(L, z)*Double_Power_Law(L, K0=K0, L_star=L_star, gamma1=gamma1, gamma2=gamma2)


def ldde_hasinger_soft(L052: np.ndarray, z: np.ndarray):
    '''
    LDDE from H05, obtained in 0.5-2 keV rest frame band
    L052 is the luminosity in 0.5-2 keV rest frame band
    z is the redshift
    '''

    return ldde(L052, z)


def ldde_hasinger_soft_cutoff(L052: np.ndarray, z: np.ndarray):
    '''
    The same as above but with exponential cutoff introduced by Brusa+2009
    '''
    if not hasattr(z, '__len__'):
        if z <= 2.7:
            return ldde_hasinger_soft(L052, z)
        else:
            return ldde_hasinger_soft(L052, 2.7)*10.**(0.43*(2.70-z)) # type: ignore
    else:
        return np.piecewise(z, [z <= 2.7, z > 2.7], [lambda z: ldde_hasinger_soft(L052, z), lambda z: ldde_hasinger_soft(L052, 2.7)*10.**(0.43*(2.70-z))])

def ldde_aird_hard(L210: np.ndarray, z: np.ndarray):
    '''
    LDDE from Aird+2010 paper, luminosity L210 is in 2-10 keV
    '''
    L = L210
    return ldde(L, z, K0=8.32e-7, L_star=10**(44.42),
                gamma1=0.77, gamma2=2.80,
                p1_44=4.64, p2_44=-1.69,
                beta1=0, beta2=0,
                z_cutoff_0=1.27,
                L_alpha=10**(44.7), alpha=0.11)


def fdpl_aird15_soft(L210, z):
    L = L210
    ksi = np.log10(1+z)

    log_K_z = -5.74+4.88*ksi-7.2*ksi**2
    log_Lstar = 43.81-0.27*ksi+6.82*ksi**2-6.94*ksi**3
    log_gamma1 = -0.18-0.55*ksi
    gamma2 = 2.43

    return 10**(log_K_z)*((L/10**log_Lstar)**(10**log_gamma1) + (L/10**log_Lstar)**gamma2)**(-1.)





class XrayLuminosityFunction():
    '''
    This is a class of an XLF with all parameters of the density are fixed, one can vary only z and L.
    '''

    def __init__(self, xlf: Callable,
                 cosmo: ccl.Cosmology = def_cosmo,
                 k_corr_po: float = 1.9,
                 m_dmh: float = 2e13,
                 name:  str = ''):
        """
        __init__ 

        xlf is an XLF of interest (it is a function of ONLY L and z, nothing else (not even logL!))
        k-correction is applied assuming a powerlaw spectrum with photon index Gamma=k_corr_po and NO absorption.
        m_dmh: mass of dark matter halo in Msun/h in which an AGN resides. This is used to calculate the bias factor of the AGN. The corresponsing bias factor is calculated from halo bias of mass m_dmh/h  M_sun (see https://www.astro.ljmu.ac.uk/~ikb/research/h-units.html), e.g. m_dmh = 1e13 -> bias for halo with mass 1/0.7~1.42e13 Msun
        name is a name of XLF for plotting and tabulating stuff

        Args:
            xlf (Callable): x-ray luminosity function
            cosmo (ccl.Cosmology, optional): cosmology to calculate fluxes, distanec, etc. Defaults to def_cosmo.
            k_corr_po (float, optional): power law slope for  K-correction. Defaults to 1.9.
            m_dmh (float, optional): mass of dark matter halo of the agn, in units if Msun/h. Defaults to 2e13.
            name (str, optional): name of the XLF. Defaults to ''.
        """


        self.xlf = xlf
        self.cosmo = cosmo
        self.k_corr_po = k_corr_po
        try:
            self.name = xlf.__name__+name
        except:
            self.name = name
        self.m_dmh = m_dmh

    def __Lz_cut__(self, logLmin: float, logLmax: float, zmin: float, zmax: float) -> Callable:
        """
        __Lz_cut__  creates XLF which is 0 outside the zmin,zmax,logLmin,logLmax region.
        It is useful because sometimes if one calculates, say, logNlogS in complex regions (with the boundaries in both z and L axes),  the L_min=4piSd(z)^2 may be less than the maximum lumunosity logLmax, and the integral from Lmin to Lmax would be negative.

        Args:
            logLmin (float): log if minimum luminosity
            logLmax (float): log if maximum luminosity
            zmin (float): minimum redshift
            zmax (float): maximum redshift

        Returns:
            Callable: XLF function which is zero outside of the given range
        """

        def lumin_cut(L):
            return 10**logLmin <= L <= 10**logLmax

        def z_cut(z):
            return zmin <= z <= zmax
        return lambda L, z: self.xlf(L, z)*lumin_cut(L)*z_cut(z)

    def plot_xlf(self, ax=None, zaxis: np.ndarray = np.linspace(0, 7, 15),
                 logLmin: float = 40, logLmax: float = 48):
        '''
        Just plots XLF in a special axis in a rangel of z values
        '''

        Laxis = np.logspace(logLmin, logLmax, 50)
        cmap = plt.get_cmap('jet', len(zaxis))
        if ax is None:
            fig, ax = plt.subplots()
        else:
            pass
        for i, zi in enumerate(zaxis):
            try:
                ax.loglog(Laxis, self.xlf(Laxis, zi), c=cmap(i))
            except:
                ax.loglog(Laxis, self.xlf(
                    Laxis, zi*np.ones(len(Laxis))), c=cmap(i))

        norm = mpl.colors.Normalize(            # type: ignore
            vmin=min(zaxis), vmax=max(zaxis))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore
        sm.set_array([])  # type: ignore
        plt.colorbar(sm, label='z', ax=ax)  # type: ignore
        ax.set_ylabel('$phi, Mpc^{-3}dex^{-1}$')
        ax.set_xlabel('$L, erg s^{-1}$')
        ax.set_title(self.name)
        return ax

    def _k_corr_(self, z: np.ndarray) -> np.ndarray:
        '''
        This is a simple function for K-correction with a single Gamma
        '''
        return k_correction_po(z, gamma=self.k_corr_po)

    def _Lmin_(self, Slim: float, z: np.ndarray) -> np.ndarray:
        """
        _Lmin_ This quantity defines the minimum luminosity your telesope would see for a given z and sensitivity Slim. It is calculated with appropriate K-correction related to the shift of energies between the observed and emitted light of an AGN.

        Args:
            Slim (float): limiting flux of a survey, NOT in units of erg/s/cm^2
            z (np.ndarray): z for which we want a limiting luminosity

        Returns:
            np.ndarray: limiting luminosity in cgs
        """

        Slim = Slim*cgs_flux
        assert Slim.unit == u.erg/u.s/u.cm**2, 'Slim should be in cgs!'  # type: ignore
        d_l = ccl.background.luminosity_distance(self.cosmo, 1/(1+z))*Mpc
        L_obs = 4*np.pi*Slim*d_l**2
        k_corr = self._k_corr_(z)  # type: ignore
        L_rest_frame = L_obs/k_corr
        return (L_rest_frame).cgs.value

    def _zmax_(self, Slim: float, L: np.ndarray) -> np.ndarray:
        """
        _zmax_ This is the same as _Lmin_(Slim,z), but from the prespective of z axis:
        solve d_L(Z)=sqrt(L/[4piS/K(Z)]) for z,
        dimensionless

        Args:
            Slim (float): limiting flux of a servey, not a unit
            L (np.ndarray): limiting luminosity

        Returns:
            np.ndarray: z_max for which the luminosity is L
        """

        Slim = Slim*cgs_flux
        L = L*cgs_lumin
        assert Slim.unit == u.erg/u.s/u.cm**2, 'Slim should be in cgs!'  # type: ignore
        assert L.unit == u.erg/u.s, 'L should be in cgs!'  # type: ignore

        zarr = np.array(optimize.fsolve(
            lambda z: 4*np.pi*Slim*(ccl.background.luminosity_distance(self.cosmo, 1/(1+z))*Mpc)**2/self._k_corr_(z) - L, 0.5))
        return zarr

    def _dVdz_(self, z: np.ndarray) -> np.ndarray:
        '''
        differential comoving volume in Mpc^3/deg^2, array of floats (not units)
        '''
        dvdzdo = differential_comoving_volume(self.cosmo, z)
        return (dvdzdo.to(Mpc**3/u.deg**2)).value  # type: ignore

    def dNdLogL(self, Slim: float = 1e-14,
                Larr: np.ndarray = np.logspace(40, 48, 75),
                cumulative: bool = False,
                zmin: float = 0.00, zmax: float = 7,
                norm: bool = False,
                plot_ax=Optional[plt.Axes], **plot_kwargs) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
        """
        dNdLogL
        This function integrates XLF in z direction so one may obatain the distribution in Luminosities for a given flux limit.
        For instance I want to know how many AGN dN have luminosity L=L0+-dL. For that I integrate:
        dN=integrate from 0 to zmax(L,Slim) XLF(L,z)d/dz * dz * dL => dN/dL= integrate from 0 to zmax(L,Slim) XLF(L,z)d/dz * dz

        The lower limit is always zmin, while the upper is the minimum of zmin(Slim,L) or zmax
        It returns the density of sources as a function of luminosity (per unit luminosity).

        Args:
            Slim (float, optional): limiting flux. Defaults to 1e-14.
            Larr (np.ndarray, optional): array of L, floats not Units. Defaults to np.logspace(40, 48, 75).
            cumulative (bool, optional): if to calculate cumulative dNdlogL. Defaults to False.
            zmin (float, optional): zmin. Defaults to 0..
            zmax (float, optional): zmax. Defaults to 7.
            norm (bool, optional): whether to normalize result at maximum. Defaults to False.
            plot_ax (Optional[plt.Axes], optional): if plt.Axes, plots the results on that axis. If none, does not plot. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: zarr and dNdLogL at zarr, units of src/deg^2
        """
        integral = np.zeros(len(Larr))
        for jj, L in tqdm(enumerate(Larr), desc='dNdL calculating', total=len(Larr), disable=True):

            a = zmin
            b = np.min([self._zmax_(Slim, L)[0], zmax])
            if a >= b:
                integral[jj] = 0
                continue
            else:
                zaxis = np.linspace(a, b, 200)
                phi = self.xlf(L, zaxis)*self._dVdz_(zaxis)
                tmp = integrate.simps(phi, zaxis)
                assert tmp > 0, f'returned integral <0! a={a}, b={b}, L={L} \n Possibly at this z, L is below the detection limit, you should never see this message!'
                integral[jj] = tmp
        if norm:
            integral = integral/np.max(integral)

        if plot_ax is None:
            pass
        else:
            plot_ax.loglog(Larr, integral, label=self.name +
                           f' \n S_lim={Slim}', **plot_kwargs)
            plot_ax.set_ylabel('dN/dlogL, $deg^{-2}$')
            plot_ax.set_xlabel('L, erg s$^{-1}$')

        if cumulative:
            tmp = integrate.cumtrapz(integral, np.log10(Larr), initial=0)
            return Larr, tmp[-1]-tmp
        return Larr, integral

    def dNdz(self, Slim: float = 1e-14,
             zarr: np.ndarray = np.concatenate(
                 (np.logspace(-3, 0.1, 25, endpoint=False), np.linspace(1, 7, 55))),
             logLmin: float = 40., logLmax: float = 48.,
             cumulative: bool = False,
             norm=False,
             Smax: float = 1e-10,
             plot_ax:Optional[plt.Axes] = None, **plot_kwargs) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
        '''
        Same as above, but for z+-dz we get dN=integrate from min[Lmin,Lmin(Slim,z)] to Lmax.
        It returns the density of sources per unit redshift
        Smax: for bin getting dNdz in some flux bin: Slim<S<Smax
        output in units of src/deg^2
        '''

        integral = np.zeros(len(zarr))

        for jj, z in tqdm(enumerate(zarr), desc='dNdz calculating', total=len(zarr), disable=True):

            a = np.max([np.log10(self._Lmin_(Slim, z)), logLmin])
            b = np.min([np.log10(self._Lmin_(Smax, z)), logLmax])
            if a >= b:
                integral[jj] = 0
                continue
            else:
                laxis = np.linspace(a, b, 200)
                phiL = self.xlf(10**laxis, z)
                tmp = integrate.simps(phiL, laxis)
                assert tmp > 0, f'returned integral={tmp[0]} <0!, z={z}.  you should never see this message!, {a=}, {b=}'
                integral[jj] = tmp*self._dVdz_(z)
        if norm:
            integral = integral/np.max(integral)

        if plot_ax is None:
            pass
        else:
            plot_ax.semilogy(zarr, integral, label=self.name +
                             f' \n {Slim}<S<{Smax}', **plot_kwargs)
            plot_ax.set_ylabel('dN/dz, $deg^{-2}$')
            plot_ax.set_xlabel('z')

        if cumulative:
            tmp = integrate.cumtrapz(integral, zarr, initial=0)
            return zarr, tmp[-1]-tmp

        return zarr, integral



    def logNlogS(self,
                 Sarr: np.ndarray = np.logspace(-15, -12, 25),
                 logLmin: float = 40., logLmax: float = 48.,
                 zmin: float = 0.05, zmax: float = 7.,
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        logNlogS calculates the number of sources per unit luminosity and per unit redshift, this is the integral of dndz over z, or dndL over L. Units are: source/deg^2/L/z.

        Args:
            Sarr (np.ndarray, optional): array of fluxes  to calculate. Defaults to np.logspace(-15, -12, 25).
            logLmin (float, optional): minimum log Luminosity. Defaults to 40..
            logLmax (float, optional): maximum log Luminosity. Defaults to 48..
            zmin (float, optional): minimal redshift. Defaults to 0.05.
            zmax (float, optional): maximum redshift. Defaults to 7..

        Returns:
            Tuple[np.ndarray, np.ndarray]: flux and source density at this flux
        """
        src_count = []
        for S in Sarr:
            zarr, dndz = self.dNdz(
                Slim=S, zarr=np.linspace(zmin, zmax, 300),
                logLmin=logLmin, logLmax=logLmax,
                norm=False,
            )
            src_count.append(integrate.simps(dndz, zarr))
        src_count = np.array(src_count)
        return Sarr, src_count




    def b_eff(self, zarr: np.ndarray, Slim: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        b_eff gets effective bias factor at given z

        Args:
            zarr (np.ndarray): z for which to get b_eff
            Slim (Optional[float], optional): Is not used for AGN. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: redshift and b_eff at this redshift
        """
        m_eff = self.m_dmh/self.cosmo['h']
        self.mdef = ccl.halos.MassDef(Delta=500, rho_type='critical')
        hbias = ccl.halos.HaloBiasTinker10(self.cosmo, mass_def=self.mdef)
        bz = np.array([hbias.get_halo_bias(self.cosmo, m_eff, a=1/(1+zi))
                       for zi in zarr])
        return zarr, bz



class ClustersXrayLuminosityFunction():
    '''
    This is a class of an XLF which uses halo model and cluster scaling relation to calculate the luminosity function of clusters in a crude way (i.e. simple selection function).
    '''

    def __init__(self,
                 cosmo: ccl.Cosmology = def_cosmo,
                 name:  str = '',
                 k_corr_file: str ='k_correction.npz'):
        """
        __init__ _summary_

        Args:
            cosmo (ccl.Cosmology, optional): cosmology for calculating distances, fluxes, etc. Defaults to def_cosmo.
            name (str, optional): name of the xlf. Defaults to ''.
            k_corr_file (str, optional): file with k-correction data, i.e.  the relation between an observed and rest-frame fluxes in different bands depending on redshift. Defaults to 'k_correction.npz'.
        """

        self.cosmo = cosmo
        self.h = cosmo['h']
        self.name = 'clusters'+name

        # WARNING: M500c  is a wrong  mass definition for such mass function. Inside MassFunction class the mass would be translated into the appropriate mass  (see https://ccl.readthedocs.io/en/latest/_modules/pyccl/halos/hmfunc.html#MassFuncTinker08) using translate_mass function (https://ccl.readthedocs.io/en/latest/_modules/pyccl/halos/massdef.html#MassDef.translate_mass)
        self.mdef = ccl.halos.MassDef(
            Delta=500, rho_type='critical')  # we use M500c as in Vikhlinin2009
        self.hmf = ccl.halos.MassFuncTinker08(cosmo,  mass_def=self.mdef)
        self.hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=self.mdef)

        # k_correction # 0.5-2 keV 
        # k_correction_0p3_2p3  #0.3-2.3 keV
        # k_correction_0p3_2p3_nh0.03 #0.3-2.3 keV, Nh=0.03*1e22
        # k_correction_0p3_2p3_nh0.06 #0.3-2.3 keV, Nh=0.06*1e22
        # k_correction_0p3_2p3_nh0.1 #0.3-2.3 keV, Nh=0.1*1e22
        # kcorr = rest_flux / obs_flux -> rest_flux = obs_flux * kcorr, but for powerlaw (in AGN) I use inverse definition, L_rest_frame = L_obs_frame/kcorr

        if k_corr_file != 'k_correction.npz':
            warn(f'k_correction file is not default, {k_corr_file}')

        k_corr_file = np.load(
            f'{rep_path}scripts/k_correction/{k_corr_file}')

        zaxis = k_corr_file['zaxis']
        kTaxis = k_corr_file['kTaxis']
        Kaxis = k_corr_file['Kaxis']
        self.apec_K_corr = interp2d(zaxis, kTaxis, Kaxis, kind='linear')

    def ML(self, M500: float, z: np.ndarray, a_lm=1.61, norm=101.483, y_lm=1.85) -> np.ndarray:
        """
        ML is a Mass-Luminosity relation from Vikhlinin et al. (2009) formula 22, see also Pillepich+ 2012, eq. (18)
        M500 - Mass is in units of Msun/h (M500c)
        z - redshift
        a_lm, norm, y_lm - parameters from Vikhlinin et al. (2009)
        """
        a = 1/(1+z)
        E = ccl.background.h_over_h0(self.cosmo, a=a)
        h = self.cosmo['h']
        lnLx = norm + a_lm*np.log(M500 / (3.9e14)) + \
            y_lm*np.log(E) - 0.39*np.log(h/0.72)
        return np.e**lnLx

    def LM(self, L: float, z: np.ndarray, a_lm: float=1.61, norm: float=101.483, y_lm: float=1.85) -> np.ndarray:
        """
        LM - this is the inverse of function ML
        """
        a = 1/(1+z)
        E = ccl.background.h_over_h0(self.cosmo, a=a)
        h = self.cosmo['h']
        M500 = 3.9e14*np.exp((np.log(L) - norm - y_lm *
                              np.log(E) + 0.39*np.log(h/0.72))/a_lm)
        return M500

    def MT(self, M500: float, z: float, a_tm: float=0.65, b_tm: float=3.02*1e14,) -> float:
        """
        MT mass-temperature relation from Vikhlinin et al. (2009).

        Args:
            M500 (float): Mass, M500c in units of Msun/h
            z (float): redshift
            a_tm (float, optional): parameter. Defaults to 0.65.
            b_tm (float, optional): parameter. Defaults to 3.02*1e14.

        Returns:
            float: Temperature for a given mass and redshift
        """

        a = 1/(1+z)
        E = ccl.background.h_over_h0(self.cosmo, a=a)
        h = self.cosmo['h']
        b_tm = b_tm/h
        lnT = a_tm*np.log(M500/b_tm)+a_tm*np.log(E) + np.log(5)
        return np.e**lnT

    def Mmin(self, Slim: float, zarr: np.ndarray, verbose: bool=False)->np.ndarray:
        """
        Mmin  this is to calculate a minimum mass Mmin (Msun) for a given flux limit Slim.

        Args:
            Slim (float): limiting flux for cluster detection, not in units
            zarr (np.ndarray): z array
            verbose (bool, optional): if True, print the M,kT,L for each z (for debugging). Defaults to False

        Returns:
            np.ndarray: Minimum mass of a cluster in given zarr
        """
        if not hasattr(zarr, "__len__"):
            zarr = np.array([zarr])
        Mmins = []
        Slim = Slim*cgs_flux

        for z in zarr:
            # !!! dont do this with broadcasting self.apec_K_corr(zarr, T_obs_arr), this would produce wrong results
            dL = ccl.background.luminosity_distance(self.cosmo, 1/(1+z))*Mpc
            Lmin_obs = 4*np.pi*Slim*dL**2
            Lmin_obs = Lmin_obs.to(cgs_lumin)
            M_min_obs = self.LM(Lmin_obs.value, z)
            T_obs = self.MT(M_min_obs, z)
            k_corr = self.apec_K_corr(z, T_obs)[0]
            Lmin_rest_frame = Lmin_obs*k_corr  # this should be "Lmin_obs*k_corr"
            M_min_rest_frame = self.LM(Lmin_rest_frame.value, z)
            T_new_mass = self.MT(M_min_rest_frame, z)
            Mmins.append(M_min_rest_frame)
            if verbose:
                print(f'Slim = {Slim}')
                print(f'{z=}')
                print(f'Lmin_obs = {Lmin_obs.value}')
                print(f'M_min_obs = {M_min_obs}')
                print(f'T_obs = {T_obs}')
                print(f'k_corr = {k_corr}')
                print(f'Lmin_rest_frame = {Lmin_rest_frame.value}')
                print(f'M_min_rest_frame = {M_min_rest_frame}')
                print(f'T_new_mass = {T_new_mass}')

        return np.array(Mmins)

    def dNdz(self, zarr: np.ndarray,
             Slim: float,
             M_min: float = 5e13,  M_max: float = 5e16,
             plot_ax: Optional[plt.Axes]=None, Smax: float = 1e-5,
             **plot_kwargs)-> Tuple[np.ndarray, np.ndarray]:
        """
        dNdz calculates dndz [deg^-2] for a given flux limit Slim.

        Args:
            zarr (np.ndarray): redshift array
            Slim (float): limiting flux for cluster detection, 
            M_min (float, optional): Minimal mass for mass function integration. Defaults to 5e13. Units: msun/h
            M_max (float, optional): Maximum mass for mass function integration. Defaults to 5e16. Units: msun/h
            plot_ax (Optional[plt.Axes], optional): if plt.Axes, plots the results on that axis. If none, does not plot. Defaults to None.
            Smax (float, optional): Max flux to integrate. Defaults to 1e-5.
            **plot_kwargs (**kwargs): keyword arguments for plotting

        Returns:
            Tuple[np.ndarray, np.ndarray]: redshift and dndz at this redshift
        """
        M_min = M_min/self.h #go to normal units: Msun/h -> Msun; e.g. was 5e13 msun/h, becomes 5e13/0.7 = 7.14e13 Msun
        M_max = M_max/self.h
        integral = np.zeros(len(zarr))
        a = 1/(1+zarr)

        for jj, z in tqdm(enumerate(zarr), desc='calculating dNdz for clusters', total=len(zarr), disable=True):

            a = np.max([self.Mmin(Slim, z)[0], M_min])
            b = np.min([self.Mmin(Smax, z)[0], M_max])
            Marr = np.geomspace(a, b, num=250)
            log10Marr = np.log10(Marr)

            if a >= b:
                integral[jj] = 0
            else:
                dndlog10m = self.hmf.get_mass_function(
                    self.cosmo, Marr, 1/(1+z), mdef_other=self.mdef)

                tmp = integrate.simps(dndlog10m, log10Marr)
                assert tmp > 0, f'returned integral={tmp} <0!, z={z}.  you should never see this message!, {a=}, {b=}'
                dens = tmp*Mpc**(-3) * \
                    differential_comoving_volume(self.cosmo, z)
                integral[jj] = dens.to(1/deg**2).value

        if plot_ax is None:
            pass
        else:
            plot_ax.semilogy(zarr, integral, label=self.name +
                             f' \n S_lim={Slim}, M_min = {M_min*self.h :2g} Msun/h', **plot_kwargs)
            plot_ax.set_ylabel('dN/dz, $deg^{-2}$')
            plot_ax.set_xlabel('z')

        return zarr, integral

    def b_eff(self, zarr: np.ndarray,
              Slim: float, M_min: float = 5e13,  M_max: float = 5e16,
              Smax: float = 1e-5)-> Tuple[np.ndarray, np.ndarray]:
        """
        the same as self.dNdz, but integrating the bias factor weighed by the mass function
        """
        M_min = M_min/self.h
        M_max = M_max/self.h
        integral = np.zeros(len(zarr))
        a = 1/(1+zarr)

        for jj, z in tqdm(enumerate(zarr), desc='calculating b_eff for clusters', total=len(zarr), disable=True):

            a = np.max([self.Mmin(Slim, z)[0], M_min])
            b = np.min([self.Mmin(Smax, z)[0], M_max])
            Marr = np.geomspace(a, b, num=250)
            log10Marr = np.log10(Marr)
            if a > b:
                integral[jj] = 0
            else:
                dndlog10m = self.hmf.get_mass_function(
                    self.cosmo, Marr, 1/(1+z), mdef_other=self.mdef)

                bm = self.hbf.get_halo_bias(
                    self.cosmo, Marr, 1/(1+z), mdef_other=self.mdef)

                tmp = integrate.simps(dndlog10m*bm, log10Marr)
                assert tmp > 0, f'returned integral={tmp} <0!, z={z}.  you should never see this message!, {a=}, {b=}'

                norm = integrate.simps(dndlog10m, log10Marr)
                w_bias = tmp/norm

                integral[jj] = w_bias
        return zarr, integral

    def logNlogS(self, Sarr: np.ndarray, zarr: np.ndarray = np.linspace(0.05, 2, 200), M_min: float = 5e13, M_max: float = 2e15,):
        """
        logNlogS calculates the number density of clusters [per deg^2] for a given flux limit by integrating the dNdz distribution for a given flux over all z.
        """
        integral = np.zeros_like(Sarr)
        for ii, Slim in enumerate(Sarr):
            zarr, dndz = self.dNdz(zarr, Slim, M_min, M_max, Smax=1)
            n_obj = integrate.simps(dndz, zarr)
            integral[ii] = n_obj

        return Sarr, integral


def_agn_xlf = XrayLuminosityFunction(xlf=ldde_hasinger_soft_cutoff,
                                     cosmo=def_cosmo,
                                     k_corr_po=1.9)


def_clusters_xlf = ClustersXrayLuminosityFunction(cosmo=def_cosmo)
