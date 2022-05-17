from scipy.signal import find_peaks
from copy import copy
import glob
import itertools 
import os
import time
import warnings
from typing import Callable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numba
import numdifftools as nd
import numpy as np
from numpy import linalg  # type: ignore
from scipy import sparse
from scipy import integrate
from scipy.special import erf
import pandas as pd
from tqdm import tqdm
import camb
import jax_cosmo
import pyccl as ccl
from camb import sources
from chainconsumer import ChainConsumer
from .utils import (make_error_boxes, path2res_forecast, rep_path)
from .luminosity_functions import XrayLuminosityFunction, ClustersXrayLuminosityFunction,  def_agn_xlf, def_clusters_xlf


sr2degsq = (np.pi/180)**-2

defaulf_cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96,
                              transfer_function='eisenstein_hu', matter_power_spectrum='linear', baryons_power_spectrum='nobaryons')


def reduce_list(t):
    """make a list of lists into a list"""
    return list(itertools.chain(*t))


def is_diag(M):
    # check if matrix is diagonal
    return (np.count_nonzero(M - np.diag(np.diagonal(M)))) == 0



def make_photoz_bin_edges(zmin: float=0.1, zmax: float=2.5, k: float=1., sigma_0: float=0.05) -> np.ndarray:
    """
    make_bin_edges creates a bin edges so that the size of bin at z_i i s is approx k*sigma_0*(1+z_i)

    Args:
        zmin (float, optional): left edge of the first bin. Defaults to 0.1.
        zmax (float, optional): approx right edge of the last bin. Defaults to 2.5.
        k (zmin, optional): approx number of photo-z widths per bin. Defaults to 1..
        sigma_0 (float, optional): sigma_0 of photo-z error. Defaults to 0.05.

    Returns:
        np.ndarray: array of left edges of bins
    """

    def sigma(z):
        if sigma_0 > 0:
            return sigma_0*(1+z)
        else:
            warnings.warn(
                'sigma_0 < zero, assume constant sigma(z) on each z')
            return -sigma_0

    if sigma_0 > 0:

        bin_edges = []
        zstart = zmin
        while zstart < zmax+k*sigma_0*(1+zmax):
            left_edge = zstart
            right_edge = left_edge + k * sigma(left_edge)
            right_edge = np.round_(right_edge, decimals=4)
            zstart = right_edge
            bin_edges.append(left_edge)

        return np.array(bin_edges)
    else:
        warnings.warn(
            'sigma_0 < zero, assume constant sigma on each z')
        return np.arange(zmin, zmax, -k*sigma_0)


def logrebin_aps(ell: np.ndarray, cls: np.ndarray, log_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    logrebin_aps rebin angular power spectrum in logarithmic manner

    Args:
        ell (np.ndarray): array of ell, in order
        cls (np.ndarray): an array of Cell values, in order
        log_bins (int): number of bins in logarithmic manner

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: mean l of a bin, cl of this bin, number of angular modes in a bin, bin width
    """
    if log_bins <= 0:
        return ell, cls, np.ones_like(ell), np.ones_like(ell)/2.
    bins = np.logspace(np.log10(ell.min()), np.log10(ell.max()), log_bins)
    bins = np.ceil(bins)
    bins = np.unique(bins)
    bins = bins[0:-1]

    digitized = np.digitize(ell, bins)

    iis = np.arange(1, np.max(digitized)+1)
    ell_bin = [np.mean(ell[digitized == i])
               for i in iis]

    cl_bin = [np.average(cls[digitized == i],
                         weights=2 * ell[digitized == i]+1)
              for i in iis]

    # calculate number of averaged spectra
    n_bin = [np.sum(digitized == i) for i in iis]
    n_bin = np.array(n_bin)
    cl_bin = np.array(cl_bin)
    l_bin = np.array(ell_bin)

    l_bin_error = n_bin / 2.

    return l_bin, cl_bin, n_bin, l_bin_error


def dNdz_photo_z(zarr: np.ndarray, dNdz: np.ndarray,
                 zmin: float = 0.5, zmax: float = 0.8,
                 f_fail: float = 0.0, sigma_0: float = 1e-3,
                 dndzname: str = '',) -> Tuple[np.ndarray, np.ndarray, float, str]:
    """
    dNdz_photo_z for a given dNdz(z)  creates a new dNdz with simulation of photo-z errors following the work of Huetsi et al. 2014, formula (7)

    Args:
        zarr (np.ndarray): input redshifts. Should be broader than the output bin due to the redistribution of z of sources after photo-z errors. Ideally - full refshift range
        dNdz (np.ndarray): redshift distribution, per deg^2
        zmin (float, optional): minimul redshift. Defaults to 0.5.
        zmax (float, optional): maximum redshift. Defaults to 0.8.
        f_fail (float, optional): fraction of catastrophic errors. Defaults to 0.0.
        sigma_0 (float, optional): accurasy of photo-z. Defaults to 1e-3.
        dndzname (str, optional): name of dNdz. Defaults to ''.


    Returns:
        Tuple: redshifts, new dNdz, number of objects in a bin, and a name of a distribution
    """
    assert np.min(
        zarr) <= zmin, 'zmin should be bigger than the minimum redshift of the input dNdz'
    assert np.max(
        zarr) >= zmax, 'zmax should be smaller than the maximum redshift of the input dNdz'
    dndzname = dndzname + \
        f' photoz: {f_fail=}, {sigma_0=}, \n {zmin}<z<{zmax}'

    if f_fail == 0 and sigma_0 == 0:
        idx = (zarr >= zmin) & (zarr <= zmax)
        dNdz_bin = dNdz
        dNdz_bin[~idx] = 0
        n_obj_bin = integrate.simpson(y=dNdz_bin, x=zarr)  # per deg^2
    else:
        n_obj = integrate.simpson(y=dNdz, x=zarr)  # per deg^2
        fz = dNdz/n_obj

        def sigma(z):
            if sigma_0 > 0:
                return sigma_0*(1+z)
            else:
                warnings.warn(
                    'sigma_0 < zero, assume constant sigma on each z')
                return -np.ones_like(z)*sigma_0

        def erf_z(z):
            assert z.shape == zarr.shape
            return erf(z/(np.sqrt(2)*sigma(zarr)))

        zmax_tot = np.max(zarr)
        fz_bin = fz*((1-f_fail) * ((erf_z(zmax - zarr) - erf_z(zmin - zarr)
                                    ) / (1 + erf_z(zarr))) + f_fail * (zmax-zmin)/zmax_tot)
        dNdz_bin = fz_bin*n_obj
        n_obj_bin = integrate.simpson(y=dNdz_bin, x=zarr)  # per deg^2

    return zarr, dNdz_bin, n_obj_bin, dndzname



########################## Fisher Matrix Class  ##############################


class FisherMatrix():
    def __init__(self, par: np.ndarray,
                 par_names: list,
                 F: np.ndarray,
                 name: str,
                 function: Callable,
                 J: np.ndarray,
                 ell_rebin: Optional[np.ndarray] = None,
                 ) -> None:
        """
        __init__ a class for handling Fisher matrices

        Args:
            par (np.ndarray): fiducial parameters of the Fisher Matrix (i.e. center of all probability contours)
            par_names (list): list of parameters, should be the same as the order of par
            F (np.ndarray): fisher matrix, should be a square symmetric matrix with size equal to the number of parameters
            name (str): name of the Fisher Matrix
            function (Callable): function which, given the parameters, returns the data  which is used to calculate the Fisher Matrix. For instance, it should return the data vector of Cells given cosmological parameters.
            J (np.ndarray): Jacobian which  is used to calculate the Fisher Matrix. 
            ell_rebin (Optional[np.ndarray], optional): values of ell for which the data vector is constructed. Defaults to None.
        """

        self.par = par
        self.par_names = par_names
        self.F = F
        self.name = name
        self.function = function
        self.J = J
        if ell_rebin is None:
            self.ell_rebin = np.arange(J.shape[0])
        else:
            self.ell_rebin = ell_rebin

    def add_prior_by_idx(self, idx, sigma, add_to_name: bool = True):
        """
        add prior to the fisher matrix
        Prior of N(mean, sigma) for parameter j is added in Fisher matrix as F[j,j]+=sigma**(-2).

        Args:
            sigmas (np.ndarray): prior sigmas

        Returns:
            FisherMatrix: new fisher matrix with added prior

        """
        sigmas = np.zeros_like(self.par)
        sigmas[idx] = sigma**(-2)
        F = self.F.copy()
        F += np.diag(sigmas)

        if add_to_name:
            newname = self.name + f' {self.par_names[idx]} prior ({sigma})'
        else:
            newname = self.name
        return FisherMatrix(self.par, self.par_names, F, newname, self.function, J=self.J, ell_rebin=self.ell_rebin)

    def rescale_by_factor(self, factor: float, add_to_name: bool = True):
        """
        returns new matrix with rescaled F by a factor
        Args:
            factor (float): rescaling factor, new F = factor*F

        Returns:
            FisherMatrix: new fisher matrix with new scale

        """
        F = self.F.copy()
        F *= factor
        if add_to_name:
            newname = self.name + f' {factor=:2g}'
        else:
            newname = self.name

        return FisherMatrix(self.par, self.par_names, F, newname, self.function, J=self.J, ell_rebin=self.ell_rebin)

    def plot_derivatives(self, idx: list[int]=[0, 1, 2]):
        """
        plot derivatives for parameters idx of the data vector used in fisher matrix calculation
        J should be set.  Plots all data derivative
        """
        if self.J is None:
            raise ValueError('J must be set')
        else:
            for i in idx:
                name = self.par_names[i]
                plt.figure(figsize=(10, 5))
                dcdpar = self.J[:, i]
                plt.plot(dcdpar, '-.')
                plt.xlabel('ell')
                plt.ylabel(f'$dCell/d{name}$')

    def plot_derivatives_2d(self, idx: list[int]=[0, 1, 2, 3, ])-> np.ndarray:
        """
        plot_derivatives_2d plots derivative but for each C_ij given that wavenumbers ell_rebin are set. 

        Args:
            idx (list[int], optional): indeces of parameters to pot. Defaults to [0, 1, 2, 3, ].

        Returns:
            np.ndarray: Jacobian with the appripriate shape: (num_of_cross_spectra, number of modes, number of parameters)
        """

        n_ell = len(self.ell_rebin)
        J_reshape = self.J.reshape(-1, n_ell, len(self.par))
        n_cls = J_reshape.shape[0]
        for par_id in idx:
            fig,  ax = plt.subplots(figsize=(8, 8))
            for bin_num in range(n_cls):
                # label = f' $dC_{{{bin_num},{bin_num}}}/d{self.par_names[par_id]}$'
                label = bin_num
                ax.semilogx(self.ell_rebin, J_reshape[bin_num, :, par_id],
                            label=label, lw=4, alpha=1/(np.sqrt(bin_num+1)), zorder=-bin_num)
                ax.set_xlabel('$\ell$')  # type: ignore
                ax.set_ylabel(
                    f'$dC/d{self.par_names[par_id]}$')
            # ax.legend()
        return J_reshape

    def check_derivatives(self, idx: list[int]=[0, 1, 2, 3, ], rel_threshold: float=0.3, title:str=''):
        """
        check_derivatives this is the helper function which checks (approximately), if the derivatives are smooth. During tests I found that sometimes derivative might have a false local peak at certain wavenumbers, which is the artifact of cosmoogica calculations. This function tries to find such peaks and inform about them.

        Args:
            idx (list[int], optional): lits of parameters to check. Defaults to [0, 1, 2, 3, ].
            rel_threshold (float, optional): threshold to declare a peak detection. Defaults to 0.3.
            title (str, optional): title of plot. Defaults to ''.

        Returns:
            bool: whether peaks are detected
        """

        deriv_ok = True
        for par_id in idx:
            deriv = self.J[:, par_id]

            peaks, _ = find_peaks(np.abs(deriv), width=[
                                  1, 2], threshold=rel_threshold*np.abs(np.max(deriv)))

            lows, _ = find_peaks(-np.abs(deriv),
                                 width=[1, 2], threshold=rel_threshold*np.abs(np.max(deriv)))
            if len(peaks) == 0 and len(lows) == 0:
                pass
            else:
                deriv_ok = False
                # print(f'{self.par_names[par_id]} has peaks and/or lows')
                # print(f'peaks: {peaks}')
                # print(f'lows: {lows}')
                warnings.warn(f'{self.par_names[par_id]} has peaks and/or lows')
                fig,  ax = plt.subplots(figsize=(15, 4))
                ax.plot(deriv, '.-.')
                ax.plot(peaks, deriv[peaks], "bo")
                ax.plot(lows, deriv[lows], "ro")
                ax.set_title(title)
                ax.set_ylabel(
                    f'$dC/d{self.par_names[par_id]}$')
            # ax.legend()
        return deriv_ok

    def __add__(self, other):
        """
        add two fisher matrices
        """
        assert np.all([self.par_names[i] == other.par_names[i] for i in range(
            len(self.par_names))]), 'par_names must be the same'
        assert self.par.shape == other.par.shape, 'par must be the same shape'
        assert self.F.shape == other.F.shape, 'F must be the same shape'

        newnamme = self.name + '+' + other.name

        F = self.F + other.F
        return FisherMatrix(par=self.par, par_names=self.par_names, F=F, name=newnamme,
                            function=self.function, J=self.J, ell_rebin=self.ell_rebin)

    def transform_to_Om_S8(self):
        """
        transform fisher matrix to the one with Omega_matter = O_cdm + O_b and S8 = sigma8(O_m/0.3)^0.5 parameters.
        See Cole 2009 for Fisher matrix transformation.

        To check, do the following:

        par_new, cov_new =F_original.par, np.linalg.inv(F_original.F)
        samples = np.random.multivariate_normal(par_new, cov_new, size=1000000)

        print((samples[:,0]+ samples[:,1]).mean())
        print((samples[:,0]+ samples[:,1]).std())
        print((samples[:,4]*np.sqrt((samples[:,0]+ samples[:,1])/0.3)).mean())
        print((samples[:,4]*np.sqrt((samples[:,0]+ samples[:,1])/0.3)).std())

        check that it coincides with the new covariance matrix after the transformation.
        """

        Oc, Ob, h, ns, s8 = self.par
        Om = Oc + Ob
        S8 = s8*(Om/0.3)**(0.5)
        M_transform = np.array([[1, -1, 0, 0, 0],
                                [0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0],
                                [-0.5*(0.3)**0.5*S8/Om**(1.5), 0, 0, 0, np.sqrt(0.3/Om)]])
        F_new = np.dot(M_transform.T, np.dot(self.F, M_transform))

        par_new = np.array([Om, Ob, h, ns, S8])
        par_new_names = ['Omega_m', 'Omega_b', 'h', 'n_s', 'S_8']

        return FisherMatrix(par_new, par_new_names, F_new, self.name, self.function, J=self.J)

    def transform_to_Om(self):
        """
        transform fisher matrix to the one with Omega_matter = O_cdm + O_b parameters.

        To check, do the following:

        par_new, cov_new =F_original.par, np.linalg.inv(F_original.F)
        samples = np.random.multivariate_normal(par_new, cov_new, size=1000000)

        print((samples[:,0]+ samples[:,1]).mean())
        print((samples[:,0]+ samples[:,1]).std())

        check that this coincide with the new covariance matrix after the transformation.
        """

        Oc, Ob, h, ns, s8 = self.par
        Om = Oc + Ob
        M_transform = np.array([[1, -1, 0, 0, 0],
                                [0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1]])
        F_new = np.dot(M_transform.T, np.dot(self.F, M_transform))

        par_new = np.array([Om, Ob, h, ns, s8])
        par_new_names = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']

        return FisherMatrix(par_new, par_new_names, F_new, self.name, self.function, J=self.J)


########################## Cell CALCULATOR FUNCTIONS  ##############################


@numba.jit
def cl_get_ij_from_idx(idx: int, n_bins: int) -> Tuple[int, int]:
    """
    cl_get_ij_from_idx Function to get the i and j indices of bins from the index of the Cl data array
    My convention is that the first positions are filled by cross-correlation of the first bin with all others (including itself). After that the same is repeated for the second bin, but excluding already-computed correlations like 1x2, for example:
    for three bins the order is:
        1x1, 1x2, 1x3, 2x2, 2x3, 3x3

    This is useful for creating data vector and covariance matrices
    initial code was:
    index_and_correlation = [(ind, x) for ind, x in enumerate(
        itertools.combinations_with_replacement(range(n_bins), 2))]
    return index_and_correlation[idx][1]
    replaced with the current code due to speed reasons

    Args:
        idx (int): index in cl data array
        n_bins (int): number of tracers (e.g. photo-z bins)

    Returns:
        tuple: (i, j): indeces of bins for which cl_data_array[idx] is computed

    """
    # n_cls = n_bins*(n_bins + 1)/2
    # assert idx < n_cls, f"idx is out of range: {idx=}>{n_cls=}"
    i = int(
        np.floor((-np.sqrt((2*n_bins+1)*(2*n_bins+1)-8*idx)+2*n_bins+1)/2))
    j = int(idx + i - i*(2*n_bins-i+1)//2)

    return i, j


@numba.jit
def cl_get_idx_from_ij(i: int, j: int, n_bins: int) -> int:
    """
    cl_get_idx_from_ij function to get the index of the Cl data array from the i and j indices of bins. Follows the same convention as cl_get_ij_from_idx.
    This is useful for creating data vector and covariance matrices
    Formula for np.triu from https://stackoverflow.com/questions/53233695/numpy-efficient-way-to-convert-indices-of-a-square-matrix-to-its-upper-triangul
    initial code was:
    check = np.vstack(np.triu_indices(n_bins)).T == np.array([i, j])
    idx = np.where(check.all(axis=1))[0][0]
    replaced with the current code due to speed reasons

    Args:
        i (int): index of the first bin
        j (int):  index of the second bin
        n_bins (int): number of tracers (e.g. photo-z bins)

    Returns:
        int: index in cl data array corresponding to cross-correlation beetween bin i and j
    """
    if j < i:
        i, j = j, i
    assert i < n_bins and j < n_bins, f"i and j are out of range "

    idx = i*(2*n_bins + 1 - i)//2 + j - i
    return idx


class DensityTracers():

    def __init__(self, zarr_list: list,
                 dndz_list: list,
                 bias_list: Optional[list] = None,
                 set_name: str = 'Density tracers',
                 ) -> None:
        """
        __init__ class to manage arrays of different LSS tracers with unique redshift distribution and bias factors

        Args:
            zarr_list (list): list of arrays of z values at which dNdz is defined
            dndz_list (list): list of arrays of dNdz, non-normalized
            bias_list (Optional[list], optional): List of bias factors at each z. If none, uses unity bias factor. If bias in a list of floats, assigns constant bias to that tracer. Defaults to None.
            set_name (str, optional): name of the set. Defaults to 'Density tracers'.
        """

        self.n_bins = len(zarr_list)
        self.labels = [f'bin_{i}' for i in range(self.n_bins)]
        self.set_name = set_name
        self.src_dens_list = [integrate.simps(
            y, x) for x, y in zip(zarr_list, dndz_list)]

        self.zarr_list = zarr_list
        self.dndz_list = dndz_list
        if bias_list is None:
            bias_list = [np.ones_like(z) for z in zarr_list]
        else:
            for i, (zarr, bias) in enumerate(zip(zarr_list, bias_list)):
                if not hasattr(bias, "__len__"):
                    bias = np.ones_like(zarr) * bias
                else:
                    pass
                bias_list[i] = bias

        self.bias_list = bias_list

    def plot_dNdz(self, savepath: Optional[str] = None, bin_left_edges: Optional[np.ndarray] = None, **plt_kwargs) -> None:
        """Plot the dn/dz data"""
        fig,  ax = plt.subplots(figsize=(8, 8))
        for i, label in enumerate(self.labels):
            label = label + f', {self.src_dens_list[i]:.2f} src/deg^2'
            ax.semilogy(self.zarr_list[i], self.dndz_list[i], '-',
                        label=label, **plt_kwargs)
            color = ax.get_lines()[-1].get_color()
            if bin_left_edges is not None:
                ax.axvspan(
                    bin_left_edges[i], bin_left_edges[i+1], color=color, alpha=0.15)
        ax.legend()
        ax.set_xlabel('z')
        ax.set_ylabel('dN/dz, deg^-2')
        ax.set_title(self.set_name)
        if savepath is not None:
            fig.savefig(savepath + 'dnndz.png')

    def make_tracers(self, cosmo_ccl: ccl.Cosmology, has_rsd: bool = False) -> Tuple:
        """
        make_tracers   (ccl cosmology objects) from an input lists of arrays.

        Args:
            cosmo_ccl (ccl.Cosmology): cosmology to calculatse kernels.
            has_rsd (bool, optional): Whether CCL tracer object have rsd. Defaults to False.

        Returns:
            Tuple: list of tracers and cosmology object
        """
        zarr_list = self.zarr_list
        dndz_list = self.dndz_list
        bias_list = self.bias_list
        trs = []
        for z, dndz, bias in zip(zarr_list, dndz_list, bias_list):
            assert len(z) == len(dndz) == len(
                bias), "z, dndz and bias must have the same length"
            tracer = ccl.NumberCountsTracer(cosmo_ccl,
                                            has_rsd=has_rsd,
                                            dndz=(z, dndz),
                                            bias=(z, bias))
            #the following line is needed  to get an easy access to the underlying z, dndz, bias and has_rsd variabeles
            #this is used to transform the ccl tracer object into a CAMB tracer object
            #see transform_ccl_tracers_to_camb for more details
            tracer.z_arr = z  # type: ignore
            tracer.dndz_arr = dndz  # type: ignore
            tracer.bias_arr = bias  # type: ignore
            tracer.has_rsd = has_rsd  # type: ignore
            trs.append(tracer)

        return trs, cosmo_ccl


def transform_ccl_tracers_to_camb(tracers: List[ccl.tracers.NumberCountsTracer]):
    """
    Transform CCL number deensity tracers to CAMB tracers.
    I ADDED ARRTIBUTES z_arr, dndz_arr, bias_arr TO ccl.NumberCountsTracer
    in function  make_tracers of class DensityTracers.

    """
    srcs = []
    for tracer in tracers:
        zarr = tracer.z_arr  # type: ignore
        dndzarr = tracer.dndz_arr  # type: ignore
        barr = tracer.bias_arr  # type: ignore
        obj = sources.SplinedSourceWindow(
            z=zarr, W=dndzarr, bias_z=barr, dlog10Ndm=0)
        srcs.append(obj)
    return srcs


def noise_Cell(src_dens_list: list) -> np.ndarray:
    """
    noise_Cell calculates (constant) noise  power and creates an array of noise ready to be added to Cell array with shape (number of cells, number of ell)
    noise in the cross power spectrum is zero
    Returns:
        np.ndarray: array of noise
    """
    n_spe = len(src_dens_list)
    idx_noise = [cl_get_idx_from_ij(i, i, n_spe)
                 for i in range(n_spe)]

    noise_levels = [1 / (sr2degsq*src_dens_list[i])
                    for i in range(n_spe)]
    noise_power = np.zeros((int(n_spe*(n_spe+1)/2)))

    noise_power[idx_noise] = noise_levels

    return noise_power


def cov_Cell(cls_rebin: np.ndarray,
             noise_power: np.ndarray,
             ell_rebin: np.ndarray,
             fsky: float,
             n_logbin: np.ndarray,
             show_progressbar: bool = True) -> np.ndarray:
    """
    cov_Cell calculates a covariance matrix for a Cell arary including cross-correlations between bins

    Formula:

    Cov(C_ij, C_kl) = normalization factor *(C_il*C_jk + C_ik*C_jl )
    normalization factor = 1/([2ell+1] * fsky * delta_ell)
    auto-spectra is with noise_power = 1/src_dens_ii

    output shape is (n_cls, n_cls, n_ell), i.e. cov[0,0,0] is the covariance between bin 0 and 0 at ell = ell[0]
    Args:
        cls_rebin (np.ndarray): input cross-correlations rebinned to ell_rebin
        noise_power (np.ndarray): Cell of noise power
        ell_rebin (np.ndarray): ell at which cls_rebin is rebinned
        fsky (float): sky fraction
        n_logbin (int): array of number of averaged bins
        show_progressbar (bool, optional): whether to show progressbar. Defaults to True.

    Returns:
        np.ndarray: covariance matrix
    """

    n_cls = cls_rebin.shape[0]
    n_bins = int(0.5*(np.sqrt(8*n_cls + 1) - 1))

    cov_norm = 1 / ((2*ell_rebin+1)*fsky*n_logbin)

    cov_rebin = np.zeros(
        (n_cls, n_cls, ell_rebin.shape[0]))
    cls_rebin_noisy = (cls_rebin.T + noise_power).T

    @numba.jit(cache=True)  # type: ignore
    def cl_get_idx_from_ij_loc(i, j):
        return i*(2*n_bins + 1 - i)//2 + j - i if i < j else j*(2*n_bins + 1 - j)//2 + i - j

    @numba.jit(cache=True)  # type: ignore
    def cl_get_ij_from_idx_loc(idx):
        i = int(np.floor((-np.sqrt((2*n_bins+1)*(2*n_bins+1)-8*idx)+2*n_bins+1)/2))
        j = int(idx + i - i*(2*n_bins-i+1)//2)
        return i, j

    for q in tqdm(range(n_cls), disable=not show_progressbar, desc='calc Covariance'):
        for w in range(q, n_cls):
            i, j = cl_get_ij_from_idx_loc(q)
            k, l = cl_get_ij_from_idx_loc(w)

            # cov (i,j), (k,l) = (i,k) * (j,l) + (i,l) * (j,k)

            # find indeces for pairs i,k; j,l; i,l; j,k
            # the functions cl_get_idx are too slow when the number of bins is large, so find index in place

            id_ik = cl_get_idx_from_ij_loc(i, k)
            id_jl = cl_get_idx_from_ij_loc(j, l)
            id_il = cl_get_idx_from_ij_loc(i, l)
            id_jk = cl_get_idx_from_ij_loc(j, k)


            cls_ik = cls_rebin_noisy[id_ik]
            cls_jl = cls_rebin_noisy[id_jl]
            cls_il = cls_rebin_noisy[id_il]
            cls_jk = cls_rebin_noisy[id_jk]

            if cls_ik[0] == cls_ik[-1]:  # to check of cl is constant
                cls_ik = np.zeros_like(cls_ik)

            if cls_jl[0] == cls_jl[-1]:
                cls_jl = np.zeros_like(cls_jl)

            if cls_il[0] == cls_il[-1]:
                cls_il = np.zeros_like(cls_il)

            if cls_jk[0] == cls_jk[-1]:
                cls_jk = np.zeros_like(cls_jk)

            res = (cls_ik * cls_jl +
                   cls_il * cls_jk)

            res = res*cov_norm

            cov_rebin[q, w] = res
            cov_rebin[w, q] = res

    return cov_rebin


def Cell_calculator(l_min: int,
                    l_max: int,
                    cosmo_ccl: ccl.Cosmology,
                    tracers: List[ccl.tracers.NumberCountsTracer],
                    log_bins: int = -1,
                    delta_i: int = -1,
                    show_progressbar: bool = True,
                    use_camb: bool = False,
                    camb_llimber: int = 110) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Cell_calculator calculates auto and cross angular power spectra given the DensityTracers object's list of tracers. It uses proper matrix sizes so that it would be easy to manage data and covariance vectors

    Args:
        l_min (int): minumim ell
        l_max (int): maximum ell
        cosmo_ccl (ccl.Cosmology): cosmology object
        tracers (List[ccl.tracers.NumberCountsTracer]): list of tracers of CCL tracer object
        log_bins (int, optional): Number of log bins of Cell/ell. If negative or zero the data is not binned. Defaults to -1.
        delta_i (int, optional): maximum bin separation (by index) after which cross-correlation is supposed to be zero. if 0, calculates only auto-correlation, if 1, calculates auto-correlation and cross-correlation with adjacent bins. If negative, calculates all correlations. Defaults to -1. Ignored spectra would result in zero blocks in the covariance matrix (see Doux 2018). Ignored idx are added to the  returned ignored_idx list.
        show_progressbar (bool, optional): whether to show progressbar. Defaults to True.
        use_camb (bool, optional): whether to use CAMB (sources) for the calculation, i.e. non-limber. Defaults to False. CAMB is faster then CCL when the number of bins is large  (>20-30)
        camb_llimber (int, optional): camb limber approx limiting l. for l>l_limber, limber approx is used by CAMB. Defaults to 110.
    Returns:
        Tuple: ell, Cell matrix,  l bin widths, ignore_idx
    """
    assert isinstance(
        tracers[0], ccl.tracers.NumberCountsTracer), f'inconsistent tracer type for CCL, got {type(tracers[0])}'

    ell = np.arange(l_min, l_max + 1)
    n_bins = len(tracers)
    n_cls = int(n_bins*(n_bins-1)/2+n_bins)
    cls = np.zeros((n_cls, len(ell)))
    ignored_idx = []

    if delta_i < 0 or delta_i > n_bins:
        delta_i = len(tracers)-1

    #this block is used if one uses CCL to calculate the spectra
    if not use_camb:
        for jj, (tr1, tr2) in tqdm(enumerate(itertools.combinations_with_replacement(tracers, 2)), disable=not show_progressbar, desc='calculating Cell', total=n_cls):
            i, j = cl_get_ij_from_idx(jj, n_bins)
            if np.abs(i-j) > delta_i:
                cls[jj] = np.zeros_like(ell)
                ignored_idx.append(jj)
            else:
                ccl_tmp = ccl.angular_cl(cosmo_ccl, tr1, tr2, ell)
                assert not np.isnan(
                    np.sum(ccl_tmp)), f'NaN in Cell calculated: {ccl_tmp=}'
                cls[jj] = ccl_tmp
    #this block is used if one uses CAMB to calculate the spectra
    else:
        cp = camb.model.CAMBparams()
        cp = _set_ccl_cosmo_to_camb_cosmo_(cosmo_ccl, cp)
        camb_tracers = transform_ccl_tracers_to_camb(tracers)
        cp.SourceWindows = camb_tracers
        cp.set_for_lmax(l_max)
        cp.SourceTerms.limber_windows = True  # type: ignore
        cp.SourceTerms.limber_phi_lmin = camb_llimber  # type: ignore
        cp.SourceTerms.counts_redshift = tracers[0].has_rsd  # type: ignore
        if show_progressbar:
            t0 = time.time()
            print('calculating CAMB...')
            results = camb.get_results(cp)
            print(f'CAMB calculation time: {time.time() - t0}')
        else:
            results = camb.get_results(cp)

        cls_res = results.get_source_cls_dict(raw_cl=True)
        ell = np.arange(2, l_max+1)
        sigma8_camb = results.get_sigma8_0()
        norm_s8 = (cosmo_ccl['sigma8'] / sigma8_camb)**2 #as we use As in camb instead of sigma8, we need to normalize the spectra to match the sigma8 of ccl
        for jj, (tr1, tr2) in enumerate(itertools.combinations_with_replacement(tracers, 2)):
            i, j = cl_get_ij_from_idx(jj, n_bins)
            if np.abs(i-j) > delta_i:
                cls[jj] = np.zeros_like(ell[ell >= l_min])
                ignored_idx.append(jj)

            else:
                camb_idx = f"W{i+1}xW{j+1}"
                ccl_tmp = cls_res[camb_idx][2:l_max+1]*norm_s8
                ccl_tmp = ccl_tmp[ell >= l_min]
                assert not np.isnan(
                    np.sum(ccl_tmp)), f'NaN in Cell calculated: {ccl_tmp=}'
                cls[jj] = ccl_tmp

        ell = ell[ell >= l_min]

    #rebin wevenumbers first
    ell_rebin, _, n_logbin, _ = logrebin_aps(ell=ell,
                                             cls=np.ones_like(
                                                 ell),
                                             log_bins=log_bins)

    cls_rebin = np.ones((n_cls, len(ell_rebin)))

    # apply logrebin_aps function for each jj in n_cls
    for jj in range(n_cls):
        cls_rebin[jj] = logrebin_aps(
            ell=ell, cls=cls[jj], log_bins=log_bins)[1]

    return ell_rebin, cls_rebin, n_logbin, ignored_idx


# @numba.jit
def sparse_arrays(cls: np.ndarray, cov: Optional[np.ndarray],
                  ignored_idx: Optional[List[int]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    sparse_arrays creates a ready-to-multiply sparse array from a covariance matrix and a Cl matrix. Deletes data and covariance from ignored_idx list.

    see https://github.com/DifferentiableUniverseInitiative/jax_cosmo/blob/master/jax_cosmo/sparse.py
    # scipy.sparse.csc_matrix
    if matrices are big, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html

    Args:
        cls (np.ndarray): Cells with shape (n_cls, n_ell)
        cov (np.ndarray): covatiance with shape (n_cls, n_cls, n_ell). If None, flattens only cls array using ignore_idx.
        ignored_idx (List[int]): indices of ignored cells: would be removed from both cls and cov, if None or empty ([]), no ignored cells are removed. Defaults to None.

    Returns:
        Tuple: new Cell and covariance matrix which are 1d and 2d respectively
    """
    cls = np.copy(cls)
    if cov is None:
        data_flatten = cls.flatten()
        if not ignored_idx:
            pass
        else:
            n_ell = cls.shape[1]
            delete_idx = [np.arange(
                n_ell*delete_idx, (delete_idx+1)*(n_ell)) for delete_idx in ignored_idx]
            data_flatten = np.delete(data_flatten, delete_idx)

        return data_flatten, None
    else:
        cov = np.array(np.copy(cov))
        assert (cov.ndim == 3) & (
            cov.shape[0] == (cov.shape[1])), f'cov must be 3d: with shape(n_cls, n_cls, ell), got {cov.shape}'
        assert cls.ndim == 2, f'cls must be 2d: with shape(n_cls, ell), got {cls.shape}'

        assert cls.shape[0] == cov.shape[
            0], f'cls and cov must have same number of rows: got {cls.shape[0]} and {cov.shape[0]}'
        assert cls.shape[1] == cov.shape[
            2], f'cls and cov must have same number of columns: got {cls.shape[1]} and {cov.shape[2]}'

        # def flatten_cov(cov):
        #     # just as jax_cosmo.sparse.to_dense(cov)
        #     cov_flattened = np.zeros(
        #         (cov.shape[0]*cov.shape[2], cov.shape[0]*cov.shape[2]))
        #     n_cls = cov.shape[0]
        #     n_ell = cov.shape[2]
        #     for ii in range(n_cls):
        #         for jj in range(n_cls):
        #             cov_flattened[ii*n_ell: ii*n_ell+n_ell, jj *
        #                           n_ell: jj*n_ell + n_ell] = np.diag(cov[ii, jj])
        #     return cov_flattened
        # cov_flatten = flatten_cov(cov)

        data_flatten = cls.flatten()
        cov_flatten = np.array(jax_cosmo.sparse.to_dense(cov))

        if not ignored_idx:
            pass
        else:
            n_ell = cov.shape[2]
            delete_idx = [np.arange(
                n_ell*delete_idx, (delete_idx+1)*(n_ell)) for delete_idx in ignored_idx]
            data_flatten = np.delete(data_flatten, delete_idx)
            cov_flatten = np.delete(cov_flatten, delete_idx, axis=0)
            cov_flatten = np.delete(cov_flatten, delete_idx, axis=1)

        return data_flatten, cov_flatten


def gaussian_loglike(data: np.ndarray, model: np.ndarray, cov: np.ndarray, include_logdet: bool = False, icov: Optional[np.ndarray] = None) -> float:
    """
    Compute the log-likelihood of a  model assuming gaussianity.

    Args:
        data (ndarray): 1d observed data
        model (ndarray): 1d model
        cov (ndarray): 2d covariance matrix
        include_logdet (bool, optional): whether to include the log-determinant of the covariance matrix. Defaults to True.
        icov (ndarray, optional): inverse of the covariance matrix. Defaults to None. If none, inverse is calculated below

    Returns:
        float: log-likelihood
    """
    assert data.shape == model.shape, \
        f'Data and model must have same shape, got {data.shape} and {model.shape}'
    assert cov.shape == (data.shape[0], data.shape[0]), \
        f'cov must have shape (n_cls*n_ell, n_cls*n_ell), got {cov.shape}'

    diff = data - model

    # check that cov does not have infs or nans
    if not np.all(np.isfinite(cov)):
        warnings.warn('WARNING: Covariance matrix contains infs or nans.')
        return -np.inf

    # we use here  inverse (inv) instead of pesudo-inveerse (pinv) because block covariance is quite easy to invert
    if icov is None:
        inv_cov = linalg.inv(cov)
    else:
        inv_cov = icov

    if include_logdet:
        sign, logdet = linalg.slogdet(cov)
        assert sign == 1.0, f'det of Covariance matrix is not positive. sign={sign}, logdet={logdet}'
        if np.all(np.isnan(logdet)):
            print('error with covar inversion, returning -inf')
            return -np.inf
    else:
        logdet = 0

    chi2 = np.einsum('i, ji,j->', diff, inv_cov, diff)

    return -0.5 * (chi2 + logdet)


class DataGenerator():

    def __init__(self,
                 fiducial_params: dict,
                 RUN_NAME: str = 'Fisher_matrices',
                 root_path_: Optional[str] = None,
                 set_name: str = 'tracers'
                 ) -> None:
        """
        __init__ this class handels the generation of datasets ready to be be used in Fisher forecast.

        Args:
            fiducial_params (dict): fiducial cosmological parameters used in data generation, including 3d power spectra P(k) parameters, neutrino masses, etc.
            RUN_NAME (str): name of the run. File should be placed in a folder with the same name. Defaults to 'Fisher_matrices'. #TODO is is not needed? I do not save anything
            root_path_ (Optional[str], optional): path to folder when different runs lie. If none, uses /inference/ folder. Defaults to None.
            set_name (str, optional): name of the set of data to generate. Defaults to 'tracers'. Good to name it as the tracer type: AGN or Clusters

        """

        if root_path_ is None:
            inference_path = path2res_forecast + 'inference'
        else:
            inference_path = path2res_forecast + root_path_
        self.RUN_NAME = RUN_NAME
        self.root_path = f"{inference_path}/{RUN_NAME}/"
        self.data_path = f'{self.root_path}data/'

        # for folder in [self.root_path, self.data_path]:
        #     if not os.path.exists(folder):
        #         os.makedirs(folder)
        #         print(f'Created folder {folder}')
        #     else:
        #         pass
        self.fiducial_params = fiducial_params
        self._cosmo_fid = ccl.Cosmology(**fiducial_params)
        self.set_name = set_name

        return None

    def gen_dNdz(self, bin_left_edges: np.ndarray,
                 f_fail: float, sigma_0: float,
                 xlf:  Union[XrayLuminosityFunction, ClustersXrayLuminosityFunction],
                 slim: float,
                 density_multiplier: float,):
        """
        gen_dNdz generates dNdz for a given XLF (X-ray luminosity function) and photo-z error parameters.

        Args:
            bin_left_edges (np.ndarray): left edges of the bins
            f_fail (float): fraction of catastrophic failures
            sigma_0 (float): redshift scatter
            xlf (Callable): XLF to integrate for dNdz function.
            slim (float, optional): Limiting flux of a survey. Float value, not in units.
            density_multiplier (float, optional): density multiplier for the dNdz function, e.g. 2 means two times more objects per sr but without alteration of dNdz.

        """
        bin_left_edges = np.array(bin_left_edges)
        self.labels = [f'bin_{i}' for i in range(len(bin_left_edges))][0:-1]
        self.bin_left_edges = bin_left_edges
        self.xlf = xlf
        self.slim = slim
        self.density_multiplier = density_multiplier
        self.f_fail = f_fail
        self.sigma_0 = sigma_0
        # if 'ldde_hasinger_soft_cutoff' in xlf.name and slim == 1e-14 and xlf.#k_corr_po == 1.9:
        #    print('load precumputed dndz for agn')
        #    zarr = np.loadtxt(
        #        rep_path+'lumin_functions/ext_data/pre_computed/AGN_zarr_1e-14_1.9.txt')
        #    dNdz = np.loadtxt(
        #        rep_path+'lumin_functions/ext_data/pre_computed/AGN_dndz_1e-14_1.9.txt')

        # if type(xlf) == ClustersXrayLuminosityFunction:
        if isinstance(xlf, ClustersXrayLuminosityFunction):
            self.type = 'Clusters'
            zarr, dNdz = xlf.dNdz(
                Slim=slim, zarr=np.linspace(0.05, 1.5, 750))
        # elif type(xlf) == XrayLuminosityFunction:
        elif isinstance(xlf, XrayLuminosityFunction):
            self.type = 'AGN'
            zarr, dNdz = xlf.dNdz(
                Slim=slim, zarr=np.linspace(0.1, 3.5, 751))
        else:
            raise ValueError(
                f'xlf must be a XrayLuminosityFunction instance, got {type(xlf)=}')

        # make more objects per deg^2. Useful to XLF which do not quite match source counts from real data or for different tests,  e.g. AGN, or for testing small sample sizes
        dNdz = dNdz * density_multiplier
        self.zarrs = []
        self.dndz_arrs = []
        for ii in range(len(self.labels)):
            zmin = bin_left_edges[ii]
            zmax = bin_left_edges[ii+1]

            zarr_phz, dNdz_phz, _, _ = dNdz_photo_z(
                zarr, dNdz, zmin=zmin, zmax=zmax, f_fail=f_fail, sigma_0=sigma_0)

            self.zarrs.append(zarr_phz)
            self.dndz_arrs.append(dNdz_phz)

        self.bin_centers = (bin_left_edges[1:] + bin_left_edges[:-1]) / 2.
        return None

    def make_tracers(self, has_rsd: bool,
                     use_weighed_bias: bool) -> None:
        """
        make_tracers make a tracer object for each bin.

        Args:
            has_rsd (bool): Whether to include RSD.
            bias_z (Union[float, np.ndarray]): bias as function of redshift. If float, interpreted as a bias of a halo of that mass. If array, interpreted as a values for bias a given zarr
            # Normally one would weight with n(z)*chi(z), but so far I use n(z) weight.
            use_weighed_bias (bool): Whether to replace b(z) with its n(z) weighted value in each bin.

        """
        zarrs = self.zarrs
        dndz_arrs = self.dndz_arrs
        self.has_rsd = has_rsd
        assert np.all([zarr == zarrs[0]
                       for zarr in zarrs]), 'zarrs are not the same!'
        if hasattr(self, 'bias_arr'):
            bias_arr = self.bias_arr
        else:
            bias_arr = self.xlf.b_eff(zarrs[0], Slim=self.slim)[1]
            self.bias_arr = bias_arr
        bias_arrs = [bias_arr for _ in zarrs]
        self.bias_arrs = bias_arrs

        nz_weighted_bias = np.array([np.average(
            bz_arr, weights=dndz_arr) for bz_arr, dndz_arr in zip(bias_arrs, dndz_arrs)])

        self.weighed_bias_arrs = nz_weighted_bias
        self.use_weighed_bias = use_weighed_bias
        if use_weighed_bias:
            # print('USING NZ WEIGHED BIASES INSTEAD OF SINGLE HALO MASS')
            bias_arrs = nz_weighted_bias
            self.bias_arrs = nz_weighted_bias
        else:
            pass

        tracers_obj = DensityTracers(
            zarrs, dndz_arrs, bias_list=list(bias_arrs), set_name=self.set_name)

        # _ = tracers_obj.get_ZData(savepath=f'{self.data_path}')
        tracers, _ = tracers_obj.make_tracers(self._cosmo_fid, has_rsd=has_rsd)

        self.n_bins = len(tracers)
        self.n_cls = int((self.n_bins)*(self.n_bins+1)/2)

        self.tracers = tracers
        self.tracers_obj = tracers_obj
        self.src_dens_list = tracers_obj.src_dens_list

    # def delta_i_feasibility(self, iis: List[int], delta_i: int):
    #     """
    #     delta_i_feasibility calculates and plots the Cells for given list of indeces and their cross-correlation with adjacend delta_i. Useful for deciding what delta_i to use.

    #     Args:
    #         iis (List[int]): List of idx to check. idx = 1 -> i,j = 0,1, etc
    #         delta_i (int): maximum bin to correlate given idx with.
    #     """
    #     for ii in iis:
    #         tracers = []
    #         src_dens_list = []
    #         for di in range(delta_i+1):
    #             tracers.append(self.tracers[ii+di])
    #             src_dens_list.append(self.src_dens_list[ii+di])
    #         tmp = Cell_calculator(
    #             l_min=10, l_max=520,
    #             cosmo_ccl=self._cosmo_fid,
    #             tracers=tracers,
    #             log_bins=26,
    #             delta_i=delta_i,
    #             use_camb=True,
    #             camb_llimber=100,
    #         )
    #         noise_power = noise_Cell(src_dens_list)

    #         ell_rebin, cls_rebin, n_logbin, _ = tmp
    #         cov_rebin = cov_Cell(cls_rebin=cls_rebin, ell_rebin=ell_rebin,
    #                              noise_power=noise_power, fsky=0.7, n_logbin=n_logbin)

    #         fig,  [ax1, ax2] = plt.subplots(2, figsize=(8, 8))  # type: ignore
    #         for di in range(delta_i+1):
    #             plot_cl(ell_rebin, n_logbin, cls_rebin, cov_rebin,
    #                     i=0, j=di, n_bins=delta_i+1,
    #                     ax=ax1, alpha=0.7/(di*1.5+1))
    #             plot_cl(ell_rebin, n_logbin, cls_rebin, cov_rebin,
    #                     i=0, j=di, n_bins=delta_i+1, plot_snr=True,
    #                     ax=ax2, alpha=0.7)

    #         ax1.legend([f'{ii}x{ii+di}' for di in range(delta_i+1)])
    #         ax2.legend([f'{ii}x{ii+di}' for di in range(delta_i+1)])
    #         # return cls_rebin

    def gen_Cell(self,
                 l_min: int, l_max: int,
                 log_bins: int, fsky: float,
                 delta_i: int = -1,
                 use_camb: bool = False,
                 camb_llimber: int = 100,
                 remove_ignored_cells: bool = True,) -> None:
        """
        gen_Cell calculates a data: Cell for a given parameters and class attibutes.

        Args:
            l_min (int, optional): l min.
            l_max (int, optional): l max.
            log_bins (int, optional): number of bins.
            fsky (float, optional): sky fraction observed.
            delta_i (int, optional): maximum bin separation (by index) after which cross-correlation is supposed to be zero. if 0, calculates only auto-correlation, if 1, calculates auto-correlation and cross-correlation with adjacent bins. If negative, calculates all correlations Defaults to -1. Ignored spectra would result in zero blocks in the covariance matrix (see Doux 2018).
            use_camb (bool, optional): Whether to use CAMB. Defaults to False.
            camb_llimber (int, optional): l_limber for CAMB. Defaults to 100.
            remove_ignored_cells (bool, optional): Whether to remove cells with zero blocks from cov and data. Defaults to True.
        """
        if use_camb and 'eisenstein_hu' in self.fiducial_params['transfer_function']:
            raise ValueError(
                'Cannot use CAMB Cell calculator with Eisenstein Hu')
        if not use_camb and 'camb' in self.fiducial_params['transfer_function']:
            warnings.warn(
                'Using CCL Cell calculator with camb power spectrum')
        if delta_i < 0 or delta_i > self.n_bins-1:
            delta_i = self.n_bins - 1
        else:
            pass
        self.delta_i = delta_i

        total_sources = np.sum(self.src_dens_list)*fsky*41253.0
        print(
            f'Total {self.type} sources: {total_sources:.0f} at {fsky=}[{fsky*41253.0:.0f} deg^2]')
        print(f'Photo-z parameters: {self.sigma_0=}, {self.f_fail=}')

        noise_power = noise_Cell(self.src_dens_list)
        tmp = Cell_calculator(l_min=l_min, l_max=l_max,
                              cosmo_ccl=self._cosmo_fid,
                              tracers=self.tracers,
                              log_bins=log_bins,
                              delta_i=delta_i,
                              use_camb=use_camb,
                              camb_llimber=camb_llimber,)
        ell_rebin, cls_rebin, n_logbin, ignored_idx = tmp
        self.ignored_idx = ignored_idx
        self.ell_rebin = ell_rebin
        self.cls_rebin = cls_rebin
        self.n_logbin = n_logbin
        self.noise_power = noise_power
        self.use_camb = use_camb
        self.camb_llimber = camb_llimber
        self.remove_ignored_cells = remove_ignored_cells
        self.l_min = l_min
        self.l_max = l_max
        self.log_bins = log_bins
        self.delta_i = delta_i

        cov_rebin = cov_Cell(cls_rebin=cls_rebin,
                             ell_rebin=ell_rebin,
                             noise_power=noise_power,
                             fsky=fsky,
                             n_logbin=n_logbin)
        if not remove_ignored_cells:
            ignored_idx = []

        cls_rebin_lkl, cov_rebin_lkl = sparse_arrays(
            cls_rebin, cov_rebin, ignored_idx)

        self.cov_rebin = cov_rebin
        self.cls_rebin_lkl = cls_rebin_lkl
        self.cov_rebin_lkl = cov_rebin_lkl

        par_vector = [self.fiducial_params['Omega_c'],
                      self.fiducial_params['Omega_b'],
                      self.fiducial_params['h'],
                      self.fiducial_params['n_s'],
                      self.fiducial_params['sigma8'],
                      ]
        par_names = ['Omega_c', 'Omega_b', 'h',  'n_s', 'sigma_8']
        # it is important to have sigma8 as the last cosmological parameter. This is because I want to calculate derivatives with respect to sigma8 analytically.

        if self.use_weighed_bias:
            _ = [par_vector.append(bias) for bias in self.weighed_bias_arrs]
            for i in range(self.n_bins):
                par_names.append(f'bias_{i+1}')
        else:
            pass
        par_vector = np.array(par_vector)
        self.par_vector = par_vector
        self.par_names = par_names
        # savepath = self.data_path
        # AllData = {'ell': ell_rebin, 'ell_error': n_logbin/2,
        #           'cls_rebin': cls_rebin_lkl, 'cov_rebin': cov_rebin_lkl}
        # pickle.dump(AllData, open(savepath+'AllData.pkl', 'wb'))

        assert self.n_cls == cls_rebin.shape[0], f'{self.n_cls=} not equal to {cls_rebin.shape[0]=}. {self.n_bins=}'

    def Cell_mean(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Cell_mean this is a function which, given cosmological parameters, calculates the output vector of data Cell with parameters exactly the same as the parameters of self.gen_Cell().

        Args:
            params (np.ndarray): cosmoloigical parameters.


        Returns:
            Tuple[np.ndarray, np.ndarray, list]: Cells data vector, Cells data vector flattened, list of indices of ignored cells. 
        """

        if not hasattr(self, 'l_min'):
            raise AttributeError(
                'Cells has not been generated yet and power spectrum parameters are not set')
        bias_idx = 5
        Oc, Ob, h, ns, s8 = params[0:bias_idx]
        # print('#### pars:', Oc, Ob, h, ns, s8)
        if self.use_weighed_bias:
            if len(params) == bias_idx:
                raise ValueError(
                    'If use_weighed_bias is True, you must provide all bias parameters.')
        if not self.use_weighed_bias:
            if len(params) != bias_idx:
                raise ValueError(
                    f'If use_weighed_bias is False, you must provide only cosmological parameters, got {params = }')

        cosmo_ccl = ccl.Cosmology(
            Omega_c=Oc, Omega_b=Ob, h=h,
            sigma8=s8,
            n_s=ns,
            transfer_function=self.fiducial_params['transfer_function'],
            matter_power_spectrum=self.fiducial_params['matter_power_spectrum'])
        if self.use_weighed_bias:
            bias_pars = params[bias_idx:bias_idx+self.n_bins]
            bias_arrs = [np.ones_like(self.zarrs[0])
                         * bias for bias in bias_pars]
        else:
            bias_arrs = self.bias_arrs

        tracers = DensityTracers(
            self.zarrs, self.dndz_arrs, list(bias_arrs))
        trs, _ = tracers.make_tracers(cosmo_ccl, has_rsd=self.has_rsd)

        _, cls_rebin, _, ignored_idx = Cell_calculator(
            l_min=self.l_min, l_max=self.l_max,
            cosmo_ccl=cosmo_ccl, tracers=trs,
            log_bins=self.log_bins,
            delta_i=self.delta_i,
            show_progressbar=False,
            use_camb=self.use_camb,
            camb_llimber=self.camb_llimber,)

        if not self.remove_ignored_cells:
            ignored_idx = []
        else:
            pass

        cls_rebin_lkl, _ = sparse_arrays(cls_rebin, None, ignored_idx)
        return cls_rebin, cls_rebin_lkl, ignored_idx

    def plot_cls(self, iis: Optional[List[int]] = None, di: Optional[int] = None,  **pltkwargs):
        """ plots cls of the datagenerator
        iis: indices of cells to plot.
        di: delta_i."""
        if di is None:
            di = self.delta_i
        else:
            pass
        if iis is None:
            iis = np.arange(self.n_bins-di+2)
        else:
            pass
        for i in iis:  # type: ignore
            fig,  ax = plt.subplots(figsize=(9, 4.5))
            for j in range(di+1):
                try:
                    _, _ = plot_cl_datagen(
                        self, i, i+j, ax=ax, alpha=0.7*1/(j+1), **pltkwargs)
                except:
                    pass

    def plot_dndzs(self, **pltkwargs):
        """ plots dndzs of the datagenerator"""
        self.tracers_obj.plot_dNdz(
            bin_left_edges=self.bin_left_edges, **pltkwargs)

    def invert_cov(self):
        """
        invert_cov inverts the covariance matrix.
        I put it into a separate function because it may be a very slow thing to calculate
        """
        try:
            data_cov_inv = self.inv_cov_rebin_lkl
        except AttributeError:
            if is_diag(self.cov_rebin_lkl):
                data_cov_inv = np.diag(1/np.diag(self.cov_rebin_lkl))
            else:
                # t0 = time.time()
                # print('Inverting covariance of data')
                data_cov_sparse = sparse.csc_matrix(self.cov_rebin_lkl)
                data_cov_inv = np.array(
                    sparse.linalg.inv(data_cov_sparse).todense())  # type: ignore
                # print(f'invert_cov took {time.time()-t0:.2f} seconds')

            self.inv_cov_rebin_lkl = data_cov_inv
            res = np.dot(data_cov_inv, self.cls_rebin_lkl)
            if np.allclose(res, np.eye(res.shape[0]), atol=1e-5, rtol=1e-5):
                warnings.warn(
                    f'inverse matrix <dot> original matrix differs from unitiy matrix!')
            self.snr = np.sqrt(np.einsum(
                'i, ji,j->', self.cls_rebin_lkl, data_cov_inv, self.cls_rebin_lkl))

        return data_cov_inv

    def get_Fisher_matrix(self, jac_step: float = 5e-4, jac_order=2,  name_postfix: str = '') -> list[FisherMatrix]:
        """
        get_Fisher_matrix function to calculate expected curvature of likelihood (Fisher matrix) assuming CONSTANT (!) covariance of data and gaussian statistics for a given DataGenerator object.

        Args:
            jac_step (float, optional): Step in ffinite difference calculation of derivative of theory vector w.r.t. cosmological parameters. Defaults to 1e-3.
            jac_order (int, optional): Order of the derivative. Defaults to 2. (two point stencil)
            name_postfix (str, optional): Postfix to add to the name of the Fisher matrix. Defaults to ''.

        Returns:
            FisherMatrix: list of FisherMatrix objects.
        """
        data_cov_inv = self.invert_cov()

        fishername = self.set_name + ''+name_postfix
        par_vector = self.par_vector
        par_names = self.par_names

        def F_func(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            F_func Fisher matrix function to calculate expected curvature of likelihood (Fisher matrix) assuming CONSTANT covariance of data and gaussian statistics for a given DataGenerator object. Inverse of this matrix is the covariance matriax of the parameters.
            It finds Fisher matrix as F = dC/dtheta^T * cov * dC/dtheta
            see also https://github.com/DifferentiableUniverseInitiative/jax-cosmo-paper/blob/master/notebooks/Fisher.ipynb
            Args:
                params (np.ndarray): fiducual parameters of cosmology to calculate fisher matrix

            Returns:
                np.ndarray: Fisher matrix evaluated with given data cov and parameters
            """
            # if self.fiducial_params['transfer_function'] == 'boltzmann_camb':
            #    warnings.warn('! Using boltzmann_camb transfer function in CCL cosmology! Check the derivative of data vector in Fisher matrix (F.plot_derivatives()) if the derivatives are continious, otherwise resulting error forecast may depend on the chosen step !')
            Cell_mean = self.Cell_mean
            cls_rebin, cls_rebin_lkl, ignored_idx = Cell_mean(params)
            if np.all(params == par_vector):
                assert np.allclose(
                    cls_rebin,  self.cls_rebin, atol=0, rtol=1e-6), 'cls_rebin from fisher at fiducial parameters is not equal to datagen.cls_rebin'

            def cell_func(x): return Cell_mean(x)[1]

            # x_cosmo are O_cdm, O_b, h, n_s. WITOUT SIGMA8
            def cell_func_cosmo(x_cosmo):
                return cell_func(reduce_list([x_cosmo, params[4:]]))

            t0 = time.time()
            print(
                f'Start Jacobian calculation (cosmological part): {jac_step=}, {jac_order=}')
            J_cosmo, info = nd.Jacobian(
                cell_func_cosmo, step=jac_step, full_output=True, method='central', order=jac_order)(params[:4])
            print(
                f'Finished Jacobian calculation (cosmological part) in {time.time()-t0:.2f} seconds')

            dCelldsigma8 = np.atleast_2d(2*cls_rebin_lkl/params[4])
            J_cosmo = np.vstack((J_cosmo.T, dCelldsigma8)).T

            if self.use_weighed_bias:
                n_bins = self.n_bins
                n_ell = self.ell_rebin.shape[0]
                n_cls = self.n_cls
                cls_rebin = self.cls_rebin

                cls_rebin_deriv_mine = np.zeros(
                    (n_cls, n_ell, len(params[5:])))

                for bias_idx in range(len(params[5:])):
                    for idx in range(n_cls):
                        if idx in ignored_idx:
                            tmp = np.zeros_like(cls_rebin[0]) - 666.
                        else:
                            i, j = cl_get_ij_from_idx(idx, n_bins)
                            if i != bias_idx and j != bias_idx:
                                tmp = np.zeros_like(cls_rebin[0])
                            elif i != j and (i == bias_idx or j == bias_idx):
                                tmp = cls_rebin[idx]/params[bias_idx+5]
                            elif i == j and (i == bias_idx or j == bias_idx):
                                tmp = 2*cls_rebin[idx]/params[bias_idx+5]
                            else:
                                tmp = np.zeros_like(cls_rebin[0])

                        cls_rebin_deriv_mine[idx, :, bias_idx] = tmp
                J_bias = cls_rebin_deriv_mine.reshape((-1, len(params[5:])))
                J_bias = J_bias[not np.all(J_bias == -666, axis=1)]
                J = np.hstack((J_cosmo, J_bias))
            else:
                J = J_cosmo
            J = np.array(J)
            # np.einsum('ik,ij,jl -> kl', J, data_cov_inv, J)
            F = np.dot(J.T, np.dot(data_cov_inv, J))

            return F, J

        def del_par(F: np.ndarray, idx: Union[int, List[int]]):
            if F.ndim == 2:
                F_new = np.delete(F, idx, axis=0)
                F_new = np.delete(F_new, idx, axis=1)
            elif F.ndim == 1:
                F_new = np.delete(F, idx)
            else:
                raise ValueError('array matrix dimension is not 1 or 2')
            return F_new

        F_fiducial, J_fiducial = F_func(par_vector)
        F_all = FisherMatrix(par=par_vector,
                             par_names=par_names,
                             F=F_fiducial,
                             name=fishername,
                             function=self.Cell_mean,
                             J=J_fiducial,
                             ell_rebin=self.ell_rebin,)

        F_no_sigma8 = FisherMatrix(par=del_par(par_vector, 4),
                                   par_names=par_names[0:4] + par_names[5:],
                                   F=del_par(F_fiducial, 4),
                                   name=fishername+'_no_sigma8',
                                   function=self.Cell_mean,
                                   J=np.delete(J_fiducial, 4, axis=1),
                                   ell_rebin=self.ell_rebin)
        F_no_sigma8_no_ns = FisherMatrix(par=del_par(par_vector, [3, 4]),
                                         par_names=par_names[0:3] +
                                         par_names[5:],
                                         F=del_par(F_fiducial, [3, 4]),
                                         name=fishername+'_no_sigma8_no_ns',
                                         function=self.Cell_mean,
                                         J=np.delete(
                                             J_fiducial, [3, 4], axis=1),
                                         ell_rebin=self.ell_rebin)

        bias_slice = np.s_[5:]
        F_no_biases = FisherMatrix(par=del_par(par_vector, bias_slice),
                                   par_names=par_names[0:5],
                                   F=del_par(F_fiducial, bias_slice),
                                   name=fishername+'_no_biases',
                                   function=self.Cell_mean,
                                   J=np.delete(J_fiducial, bias_slice, axis=1),
                                   ell_rebin=self.ell_rebin)

        return [F_all, F_no_sigma8, F_no_sigma8_no_ns, F_no_biases]

    def invoke(self,
               bin_left_edges: np.ndarray,
               l_min: int = 25, l_max: int = 520, log_bins: int = 31,
               f_fail: float = 0.1, sigma_0: float = 0.05,
               xlf: XrayLuminosityFunction = def_agn_xlf,
               slim: float = 1e-14,
               has_rsd: bool = False,
               fsky: float = 0.7,
               delta_i: int = -1,
               use_camb: bool = True,
               camb_llimber: int = 100,
               use_weighed_bias: bool=False,
               density_multiplier: float = 1,
               plot_dndz: bool = True,
               plot_cell: bool = True,
               remove_ignored_cells: bool = True,
               calc_cl: bool = True,
               ):
        """
        invoke invoke this object to create data and prepare for the fitting


        Args:
            bin_left_edges (np.ndarray): left edges of the bins
            l_min (int, optional): l min - minumum angular wavenumber \ell. Defaults to 30.
            l_max (int, optional): l max - maximum angular wavenumber \ell. Defaults to 500.
            log_bins (int, optional): number of \ell bins. Defaults to 20.
            f_fail (float, optional): fraction of catastrophic errors in photo-redshift determination . Defaults to 0.1.
            sigma_0 (float, optional): redshift scatter of photo-z errors: sigma(z)=sigma_0*(1+z). Defaults to 0.05.
            xlf (XrayLuminosityFunction, optional): X-Ray luminosity function to integrate. Defaults to def_agn_xlf.
            slim (float, optional): Limiting flux of the survey. Defaults to 1e-14.
            has_rsd (bool, optional): whether to take into account the redshift-space distortion. Defaults to False.
            fsky (float, optional): sky fraction observed. Defaults to 0.7.
            delta_i (int, optional): maximum bin separation (by index) after which cross-correlation is supposed to be zero. if 0, calculates only auto-correlation, if 1, calculates auto-correlation and cross-correlation with adjacent bins. If negative, calculates all correlations Defaults to -1. Ignored spectra would result in zero blocks in the covariance matrix (see Doux 2018).. Defaults to -1.
            use_camb (bool, optional): Whether to use CAMB.. Defaults to False.
            camb_llimber (int, optional): l_limber for CAMB. Defaults to 100.
            use_weighed_bias (bool, optional): whether to use constant b(z) for a bin, with the real b(z) weighted with dndz of a bin. Defaults to False.
            density_multiplier (float, optional): density multiplier for the xlf. Defaults to 1.
            plot_dndz (bool, optional): whether to plot dndz of the data generator. Defaults to True.
            plot_cell (bool, optional): whether to plot Cell of the data generator. Defaults to True.
            remove_ignored_cells (bool, optional): Whether to remove cells with zero blocks from cov and data. Defaults to True.
            calc_cl (bool, optional): Whether to calculate the power spectra. Defaults to True.


        """

        self.l_min = l_min
        self.l_max = l_max
        self.log_bins = log_bins
        self.fsky = fsky
        self.delta_i = delta_i

        self.gen_dNdz(bin_left_edges=bin_left_edges,
                      f_fail=f_fail, sigma_0=sigma_0,
                      xlf=xlf, slim=slim,
                      density_multiplier=density_multiplier)

        self.make_tracers(has_rsd=has_rsd,
                          use_weighed_bias=use_weighed_bias)
        if plot_dndz:
            self.plot_dndzs(lw=3, alpha=0.7)
        if calc_cl:
            self.gen_Cell(l_min=l_min, l_max=l_max, log_bins=log_bins,
                          fsky=fsky, delta_i=delta_i, use_camb=use_camb, camb_llimber=camb_llimber,
                          remove_ignored_cells=remove_ignored_cells)
            # self.invert_cov()
            if plot_cell:
                self.plot_cls(iis=[0, self.n_bins//2, self.n_bins-1])

        return None




def plot_cl(ell_rebin: np.ndarray, 
            n_logbin: np.ndarray, 
            cls_rebin: np.ndarray, 
            cov_rebin: np.ndarray, 
            i: int, j: int, n_bins: int,
            ax: Optional[plt.Axes] = None, 
            lw: float=4, alpha: float=0.6, label:  str='',
            plot_snr: bool = False, shot_noise: float = 0,
            addlabel: bool = True,
            **plot_kwargs):
    """
    plot_cl plot the angular power spectrum

    Args:
        ell_rebin (np.ndarray): input ell bins
        n_logbin (np.ndarray): number of modes in each bin
        cls_rebin (np.ndarray): input cls
        cov_rebin (np.ndarray): input covariance
        i (int): index to plot cell of, e.g i=1, j=2 plots C_12
        j (int): as i
        n_bins (int): number of redshift bins
        ax (Optional[plt.Axes], optional): axis to plot, if None, axis is created. Defaults to None.
        lw (float, optional): line width. Defaults to 4.
        alpha (float, optional): alpha. Defaults to 0.6.
        label (str, optional): label. Defaults to ''.
        plot_snr (bool, optional): whether to plot Cell/sigma(Cell). Defaults to False.
        shot_noise (float, optional): level of shot noise, if 0, does not plot noise. Defaults to 0.
        addlabel (bool, optional): whether to add label to the  plot. Defaults to True.
        **plot_kwargs (**kwargs): keyword arguments to pass to plt.plot

    Returns:
         fig and ax
    """

    idx = cl_get_idx_from_ij(i, j, n_bins)
    if ax is None:
        fig,  ax = plt.subplots(figsize=(8, 8))
    else:
        ax = ax
        fig = plt.gcf()
    if addlabel:
        label += f' $C_{{{i},{j}}}$'
    else:
        pass
    if not plot_snr:
        make_error_boxes(ax, ell_rebin, cls_rebin[idx], n_logbin/2,
                         cov_rebin[idx, idx]**0.5, lw=lw, alpha=alpha, label=label, **plot_kwargs)
        ax.set_ylabel(r'$C_\ell\,  {\rm [sr]}$', fontsize=25)
        color = ax.get_lines()[-1].get_color()
        ax.set_xlim()
        ax.set_ylim()
        if shot_noise != 0:
            #ylim = ax.get_ylim()
            #xlim = ax.get_xlim()
            ax.axhline(shot_noise, color=color, ls='-.', lw=5, alpha=alpha)
            # ax.set_ylim(ylim)
            # ax.set_xlim(xlim)
    else:
        ax.semilogx(ell_rebin, cls_rebin[idx]/cov_rebin[idx, idx]
                    ** 0.5, label=label, lw=lw, alpha=alpha, **plot_kwargs)
        ax.set_ylabel(r'$C_\ell/\sigma_\ell$')

    ax.set_xlabel(r'$\ell$', fontsize=25)
    ax.legend(fontsize=20)

    return fig, ax


def plot_cl_datagen(datagen: DataGenerator, i: int, j: int,  ax: Optional[plt.Axes] = None, lw: float=4, alpha: float=0.6, label:str='', **plot_kwargs):
    """
    plot_cl_datagen calls plot_cl on the datagenerator
    """
    idx = cl_get_idx_from_ij(i, j, datagen.n_bins)
    shot_noise = datagen.noise_power[idx]
    fig, ax = plot_cl(datagen.ell_rebin,
                      datagen.n_logbin,
                      datagen.cls_rebin,
                      datagen.cov_rebin,
                      i, j, datagen.n_bins, ax=ax, lw=lw, alpha=alpha, label=label, shot_noise=shot_noise, **plot_kwargs)

    ax.set_title(datagen.set_name)
    return fig, ax


def fisher_summary(Fs: List[FisherMatrix], par_idx: Optional[List[int]], precision: int = 3, factor: float=1.0) -> list:
    """
    fisher_summary makes a summary of the Fisher matrices, i.e. parameters, their errors, snr, FoM value etc.

    Args:
        Fs (List[FisherMatrix]): List of Fisher matrices
        par_idx (Optional[List[int]]): index  of parameters to report. 
        precision (int, optional): precision . Defaults to 3.
        factor (float, optional): factor to multiply matrices. Defaults to 1.0.

    Returns:
        list: _description_
    """
    list_of_lists = []
    for F in Fs:
        list = []
        print(F.name)
        cov = linalg.inv(F.F*factor)
        errors = np.sqrt(np.diag(cov))
        snr = 100*errors/F.par
        FoM = np.pi/np.sqrt(linalg.det(cov))
        # FoM = np.log10(1/np.sqrt(linalg.det(cov)))
        print(f'FoM: {FoM:.{precision}g}')
        if par_idx is None:
            par_idx = np.arange(len(F.par_names))
        for i, par_name in enumerate(F.par_names):
            # if 'bias' in par_name and not print_biases:
            #    continue
            message = f'{par_name}: {F.par[i]:.{precision}f} +- {errors[i]:.{precision}f} ({snr[i]:.1f} %)'
            print(message)
            list.append([message])
        # if not print_biases:
        #    print('biases are not shown')
        print('-------')
        list_of_lists.append(list)
    return list_of_lists


def compare_fisher_matrices(Fs: list[FisherMatrix], 
                            title: str='', filename: Optional[str]=None,  
                            names_list: Optional[list[str]]=None, 
                            factor: float=1., fsky: Optional[float]=None, 
                            figsize: float=10, plot_table: bool=True, precision: int=3,  **config_kw):
    """
    compare_fisher_matrices plots and compares Fisher matrices.

    Args:
        Fs (list[FisherMatrix]): list of Fisher matrices
        title (str, optional): title of the plot. Defaults to ''.
        filename (Optional[str], optional): name of saved pot. Defaults to None.
        names_list (Optional[list[str]], optional): list of names of fisher matrices (to replace F.name). Defaults to None.
        factor (float, optional): factor to multiply matrices. Defaults to 1..
        fsky (Optional[float], optional): fsky (for title). Defaults to None.
        figsize (float, optional): plot arg. Defaults to 10.
        plot_table (bool, optional): whether to plot a parameters table on the plot. Defaults to True.
        precision (int, optional): precision for table. Defaults to 3.

    Returns:
        figure and table
    """
    #factor = fsky/0.658
    par_idx = [0, 1, 2, 3, 4]
    ch_cons = ChainConsumer()
    par_names = [Fs[0].par_names[ii] for ii in par_idx]
    par_names = [x.replace('Omega', '\Omega') for x in par_names]
    par_names = [x.replace('sigma', '\sigma') for x in par_names]
    par_names = ['$'+x+'$' for x in par_names]
    par_names_fom = copy(par_names)
    par_names_fom.append('FoM')

    for jj, F in enumerate(Fs):
        F = copy(F)

        if names_list is None:
            pass
        else:
            F.name = names_list[jj]
        F_matr = F.F*factor
        cov = np.linalg.inv(F_matr)  # type: ignore
        cov = cov[par_idx, :][:, par_idx]
        par = F.par[par_idx]

        ch_cons.add_covariance(
            par, cov, parameters=par_names, name=f'{jj+1}: '+F.name)

    ch_cons.configure(usetex=False, serif=False,
                      sigma2d=False, summary=True, bar_shade=True, **config_kw)

    fig_chains = ch_cons.plotter.plot(
        figsize=(figsize, figsize), filename=None)
    if fsky is not None:
        fig_chains.suptitle(ttl := title+f'\n$f_{{\\rm sky}} = {fsky}$',)
    else:
        fig_chains.suptitle(ttl := title)

    table = pd.DataFrame()
    for F in Fs:
        F_matr = F.F*factor
        cov = np.linalg.inv(F_matr)
        snrs = list(np.sqrt(np.diag(cov))/F.par*100)
        snrs = ['{:.1f}%'.format(x) for x in snrs]
        fom = np.log10(np.pi/np.sqrt(np.linalg.det(cov)))
        fom = '{:.2f}'.format(fom)
        snrs.append(fom)
        table = pd.concat((table, pd.DataFrame([snrs], columns=par_names_fom))) #table.append(pd.DataFrame([snrs], columns=par_names_fom))
    table.index = [F.name for F in Fs]
    # if not agn_clu_comb:
    # else:
    #    table.index = ['Clusters', 'AGN', 'Comb.']
    cell_text = table.values
    if plot_table:
        tab = fig_chains.axes[8].table(cellText=cell_text,
                                       colLabels=['$\delta$'+x+'/' +
                                                  x for x in table.columns[0:-1]]+['FoM'],
                                       rowLabels=list(range(1, len(Fs)+1)),
                                       bbox=[-0.25, 0, 2.5, 1],
                                       colWidths=[0.25]*len(table.columns),
                                       rowColours=[x.get_color() for x in fig_chains.axes[0].get_lines()])
        tab.auto_set_font_size(False)
        tab.set_fontsize(12)
        tab.scale(2,  1.5)
    else:
        pass
    if filename is not None:
        plt.savefig(filename)
    print(ttl)
    list_of_lists = fisher_summary(
        Fs=Fs, par_idx=None, factor=factor,
        precision=precision)
    print('======')
    return fig_chains, table


def _set_ccl_cosmo_to_camb_cosmo_(cosmo: ccl.Cosmology, cp):
    """
    _set_ccl_cosmo_to_camb_cosmo_ sets camb cosmology to ccl cosmology.

    Args:
        cosmo (ccl.Cosmology): ccl cosmology
        cp camb parameterss dictionary

    Returns:
        camb parameterss dictionary
    """
    from pyccl import ccllib as lib
    from pyccl.pyutils import check

    # =====  BELOW IS THE SETTING OF CAMB COSMOLOGY TO THE INPUT CCL COSMOLOGY ===== #
    # see https://ccl.readthedocs.io/en/latest/_modules/pyccl/boltzmann.html#get_camb_pk_lin
    # import camb
    extra_camb_params = {}
    if cosmo._config_init_kwargs['matter_power_spectrum'] != 'linear':
        warnings.warn('WARNING: CCL cosmoology has non-linear power spectrum!')
    if cosmo._config_init_kwargs['transfer_function'] != 'boltzmann_camb':
        warnings.warn(
            'CCL cosmology has non-CAMB transfer function and we tried to build CAMB cosmology on that! Results may be edifferent is compared to (Limber) CCL calculations')

    # z sampling from CCL parameters
    na = lib.get_pk_spline_na(cosmo.cosmo)
    status = 0
    a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
    check(status)
    a_arr = np.sort(a_arr)
    zs = 1.0 / a_arr - 1
    zs = np.clip(zs, 0, np.inf)

    if np.isfinite(cosmo["A_s"]):
        A_s_fid = cosmo["A_s"]
    elif np.isfinite(cosmo["sigma8"]):
        # in this case, CCL will internally normalize for us when we init
        # the linear power spectrum - so we just get close
        A_s_fid = 2.43e-9 * (cosmo["sigma8"] / 0.87659)**2
    else:
        raise ValueError("Must provide either A_s or sigma8")

    # calculate as_fid
    # see https://cosmocoffee.info/viewtopic.php?t=475
    # cosmo_ccl_as = ccl.Cosmology(Omega_c=cosmo['Omega_c'], Omega_b=cosmo['Omega_b'], h=cosmo['h'], A_s=A_s_fid, sigma8=None,  n_s=cosmo['n_s'], transfer_function='boltzmann_camb', matter_power_spectrum=cosmo._config_init_kwargs['matter_power_spectrum'], baryons_power_spectrum=cosmo._config_init_kwargs['baryons_power_spectrum'])

    # ratio = cosmo['sigma8']/ccl.sigma8(cosmo_ccl_as)
    # A_s_fid = A_s_fid * ratio**2

    # init camb params
    # cp = camb.model.CAMBparams()

    # turn some stuff off
    cp.WantCls = True
    cp.DoLensing = False
    cp.Want_CMB = False
    cp.Want_CMB_lensing = False
    cp.Want_cl_2D_array = True
    cp.WantTransfer = True

    # basic background stuff
    h2 = cosmo['h']**2
    cp.H0 = cosmo['h'] * 100
    cp.ombh2 = cosmo['Omega_b'] * h2
    cp.omch2 = cosmo['Omega_c'] * h2
    cp.omk = cosmo['Omega_k']

    # "constants"
    cp.TCMB = cosmo['T_CMB']

    # neutrinos
    # We maually setup the CAMB neutrinos to match the adjustments CLASS
    # makes to their temperatures.
    cp.share_delta_neff = False
    cp.omnuh2 = cosmo['Omega_nu_mass'] * h2
    cp.num_nu_massless = cosmo['N_nu_rel']
    cp.num_nu_massive = int(cosmo['N_nu_mass'])
    cp.nu_mass_eigenstates = int(cosmo['N_nu_mass'])

    delta_neff = cosmo['Neff'] - 3.046  # used for BBN YHe comps

    # CAMB defines a neutrino degeneracy factor as T_i = g^(1/4)*T_nu
    # where T_nu is the standard neutrino temperature from first order
    # computations
    # CLASS defines the temperature of each neutrino species to be
    # T_i_eff = TNCDM * T_cmb where TNCDM is a fudge factor to get the
    # total mass in terms of eV to match second-order computations of the
    # relationship between m_nu and Omega_nu.
    # We are trying to get both codes to use the same neutrino temperature.
    # thus we set T_i_eff = T_i = g^(1/4) * T_nu and solve for the right
    # value of g for CAMB. We get g = (TNCDM / (11/4)^(-1/3))^4
    g = np.power(
        lib.cvar.constants.TNCDM / np.power(11.0/4.0, -1.0/3.0),
        4.0)

    if cosmo['N_nu_mass'] > 0:
        nu_mass_fracs = cosmo['m_nu'][:cosmo['N_nu_mass']]
        nu_mass_fracs = nu_mass_fracs / np.sum(nu_mass_fracs)

        cp.nu_mass_numbers = np.ones(cosmo['N_nu_mass'], dtype=int)
        cp.nu_mass_fractions = nu_mass_fracs
        cp.nu_mass_degeneracies = np.ones(int(cosmo['N_nu_mass'])) * g
    else:
        cp.nu_mass_numbers = []
        cp.nu_mass_fractions = []
        cp.nu_mass_degeneracies = []

    # get YHe from BBN
    cp.bbn_predictor = camb.bbn.get_predictor()  # type: ignore
    cp.YHe = cp.bbn_predictor.Y_He(
        cp.ombh2 * (camb.constants.COBE_CMBTemp /  # type: ignore
                    cp.TCMB) ** 3,
        delta_neff)

    camb_de_models = ['DarkEnergyPPF', 'ppf', 'DarkEnergyFluid', 'fluid']
    camb_de_model = extra_camb_params.get('dark_energy_model', 'fluid')
    if camb_de_model not in camb_de_models:
        raise ValueError("The only dark energy models CCL supports with"
                         " camb are fluid and ppf.")
    cp.set_classes(
        dark_energy_model=camb_de_model
    )

    if camb_de_model not in camb_de_models[:2] and cosmo['wa'] and \
            (cosmo['w0'] < -1 - 1e-6 or
                1 + cosmo['w0'] + cosmo['wa'] < - 1e-6):
        raise ValueError("If you want to use w crossing -1,"
                         " then please set the dark_energy_model to ppf.")
    cp.DarkEnergy.set_params(
        w=cosmo['w0'],
        wa=cosmo['wa']
    )

    nonlin = False
    cp.set_matter_power(
        redshifts=[_z for _z in zs],
        kmax=extra_camb_params.get("kmax", 10.0),
        nonlinear=nonlin)
    cp.set_matter_power(nonlinear=nonlin)
    assert cp.NonLinear == camb.model.NonLinear_none

    cp.InitPower.set_params(
        As=A_s_fid,  # type: ignore
        ns=cosmo['n_s'])  # type: ignore

    assert cp.omk == 0, 'Non-flat cosmology for CAMB!'
    return cp


def make_cobaya_input(datagen: DataGenerator, foldername: str, F: FisherMatrix, fix_cov: bool,
                      filename: str='info_auto.yaml',):
    """
    make_cobaya_input makes a yaml file for cobaya to use in MCMC fitting.

    Args:
        datagen (DataGenerator): data generator object for which you want to run MCMC
        foldername (str): folder where to save data and all files
        F (FisherMatrix): Fisher matrix to get the initial covariance matrix
        fix_cov (bool): whether to fix the covariance matrix or not
        filename (str, optional): name of cobaya input file. Defaults to 'info_auto.yaml'.
    """

    path = path2res_forecast + 'mcmc_inference/' + foldername
    for folder in [path]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f'Created folder {folder}')

    if F is not None:
        F = F.F
        np.savetxt(path + '/fisher_cov.txt', np.linalg.inv(F),
                   header=' '.join(['Omega_c', 'Omega_b', 'h', 'n_s', 'sigma8']))
    np.savetxt(path + '/data_vector.txt', datagen.cls_rebin_lkl)
    np.savetxt(path + '/bin_left_edges.txt', datagen.bin_left_edges)

    import yaml
    info_auto = {
        'params':
            {'Omega_c':
                {'prior': {'min': 0.0, 'max': 0.8},
                    'proposal': 0.02,
                    'ref': {'min': 0.23, 'max': 0.27}},

                'Omega_b':
                {'prior': {'min': 0.001, 'max': 0.3},
                    'proposal': 0.01,
                    'ref': {'min': 0.045, 'max': 0.055}},

                'h':
                {
                    'prior': {'dist': 'norm',
                              'loc': float(datagen.fiducial_params['h']), 'scale': 0.05},
                    'proposal': 0.02,
                    'ref': {'min': 0.68, 'max': 0.72}},

                'n_s':
                {
                    'prior': {'dist': 'norm',
                              'loc': float(datagen.fiducial_params['n_s']), 'scale': 0.01},
                    'proposal': 0.01,
                    'ref': {'min': 0.94, 'max': 0.98}},

                'sigma8':
             {'prior': {'min': 0.6, 'max': 1.0},
                    'proposal': 0.005,
                    'ref': {'min': 0.78, 'max': 0.82},
              }},


        'theory':
        {'forecast.cobaya_classes.CCLclustering':
                {
                    'type': datagen.type,
                    'f_fail': datagen.f_fail,
                    'sigma_0': datagen.sigma_0,
                    'l_min': datagen.l_min,
                    'l_max': datagen.l_max,
                    'log_bins': datagen.log_bins,
                    'fsky': datagen.fsky,
                    'slim': datagen.slim,
                    'delta_i': datagen.delta_i,
                    'use_camb': datagen.use_camb,
                    'camb_llimber': datagen.camb_llimber,
                    'has_rsd': datagen.has_rsd,
                    'use_weighed_bias': datagen.use_weighed_bias,
                    'density_multiplier': datagen.density_multiplier,
                    'remove_ignored_cells': datagen.remove_ignored_cells,
                    'bin_left_edges_file':  './bin_left_edges.txt',
                    'transfer_function': datagen._cosmo_fid._config_init_kwargs['transfer_function'],
                    'matter_pk': datagen._cosmo_fid._config_init_kwargs['matter_power_spectrum'],
                    'baryons_pk': datagen._cosmo_fid._config_init_kwargs['baryons_power_spectrum'],
                    'fix_cov': fix_cov, }},
        'likelihood':
        {'forecast.cobaya_classes.GaussianClLikelihood':
                {'data_vector_file': './data_vector.txt',
                 'fix_cov': fix_cov, }},
        'sampler':
        {'mcmc':
                {'Rminus1_stop': 0.02,
                    'burn_in': 0,
                    'max_samples': 5000,
                    'output_every': '60s',
                    'learn_every': 250,
                    'learn_proposal': True,
                    'measure_speeds': True,
                    'covmat': './fisher_cov.txt',
                 }
         },
        'output': 'chains/chain',
        'debug': False}

    # for i, bias in enumerate(datagen.weighed_bias_arrs, 1):
    #     print(i, bias)
    #     minbi = float(bias*0.9)
    #     maxbi = float(bias*1.1)
    #     # float as simpler than numpy stuff from bias, and can be saved in yaml

    #     info_auto['params'][f'bias_{i}'] = {'prior': {'min': 0.7, 'max': 10.0}, 'ref': {
    #         'min': minbi, 'max': maxbi}, 'drop': True, 'proposal': 0.01}

    # bias_str = ', '.join(f'bias_{i}' for i in range(
    #     1, len(datagen.weighed_bias_arrs)+1))
    # bias_str = f'lambda {bias_str}: [{bias_str}]'

   # info_auto['params']['bias_vector'] = {
   #     'value': bias_str, 'derived': False}

    with open(f'{path}/{filename}', 'w') as file:
        yaml.dump(info_auto, file, sort_keys=False)

    with open(f'{path}/readme.txt', 'w') as file:
        str = """
    CHECK PARAMETERS:  WHETHER TO CALCULATE CROSS CORRELATION,  FSKY, NUMBER OF BINS AND THEIR RANGE
    CHECK TRANSFER FUNCTION IN ACCORDANCE WITH THE DATA GENERATOR (if you wish)
    RUN WITH
    CHECK PRIORS OF YOU DO NOT USE STANDARD COSMOLOGY (0.25, 0.05, 0.7)
    run test:
    cobaya-run --test  info_auto.yaml
    and then run cobaya:
    mpirun -n 8 cobaya-run -r  info_auto.yaml >> log.txt &

    sge runs:
    qsub sge.txt
    check that #$ is used instead of # $

        """
        file.write(str)
        file.close()

    # os.system(f'mkdir -p {datagen.root_path}/sge')
    with open(f'{path}/sge.txt', 'w') as file:
        str = f"""
#$ -S /bin/bash
#$ -cwd
#$ -e sge/stderr
#$ -o sge/stdout
#$ -m beas
#$ -M sdbykov@mpa-garching.mpg.de
#$ -l h_cpu=48:00:00
#$ -l h_vmem=80000M
#$ -N clustering_mcmc
#$ -pe mpi 8


module load anaconda3
module load openmpi
module list
source /afs/mpa/home/sdbykov/.bashrc


mpirun -n 8 cobaya-run -r  info_auto.yaml >> log.txt


        """
        file.write(str)
        file.close()


def analyze_chain(SETNAME,
                  burn_in_fraction: float = 0.1,
                  chain_name: Optional[str] = None,
                  figsize: int=10,
                  filename: Optional[str]=None,
                  Fs: Optional[List[FisherMatrix]] = None,
                  **config_kw):
    """
    analyze_chains analyzes mcmc chains, compares with fisher matrix, and plots the results.

    Args:
        SETNAME: name of the dataset to analyze, from the folder 'results/data/mcmc_inference/'
        burn_in_fraction (float, optional): Burn in fraction of all samples. Defaults to 0.1.
        chain_name (str, optional): If given, retrieves specific chain from the chains/ folder. Else uses all chains. Defaults to None.
        figsize (int, optional): Size of the figure. Defaults to 10.
        filename (str, optional): If given, saves the figure to the file. Defaults to None.
        Fs (List[FisherMatrix], optional): List of fisher matrix objects to plot. Defaults to None. Used if you want to compare Fisher forecast and MCMC fit.


    Returns:
        Tuple: dataframe with results from all chains, all chains separably, and chain consumer object
    """
    chains_path = f"{rep_path}/forecast/mcmc_inference/{SETNAME}/chains/"

    os.chdir(chains_path)
    ch_cons = ChainConsumer()

    par_names = ['$\Omega_c$',
                 '$\Omega_b$', '$h$', '$n_s$', '$\sigma_8$']
    if Fs is not None:
        for F in Fs:
            cov = np.linalg.pinv(F.F)  # type: ignore
            par = F.par
            par_names = par_names  # F.par_names

            ch_cons.add_covariance(
                par, cov, parameters=par_names, name='Fisher')

    if chain_name is None:
        files = glob.glob(chains_path + '/*.txt')
    else:
        files = [chains_path + '/' + chain_name + '.txt']

    chains_list = []

    if len(files) == 0:
        raise Exception('No chains found')

    for file in files:
        # print('################')
        #print(f' working with {file}')
        f = open(file, 'r')
        line1 = f.readline()
        chain_res = pd.DataFrame(pd.read_csv(f, sep='\s+',  # type: ignore
                                             names=line1.replace('#', ' ').split()))
        # print('best fit in this file:')
        # print(chain_res.loc[chain_res['minuslogpost'].idxmin()])
        cols = [c for c in chain_res.columns if not (('minuslogprior' in c) or (
            'minuslogpost' in c) or ('chi2' in c) or ('weight' in c))]
        #cols = [x.replace('Omega', '\Omega') for x in cols]
        #cols = [x.replace('sigma', '\sigma') for x in cols]
        #cols = ['$'+x+'$' for x in cols]
        # tmp_cols = [cols[1], cols[0], cols[-1]]
        # _ = [tmp_cols.append(x) for x in cols[2:-1]]
        # tmp_cols = [cols[1], cols[0], cols[2]]
        # _ = [tmp_cols.append(x) for x in cols[3:]]
        # cols = tmp_cols
        chain_res = chain_res[cols]

        chain_res = chain_res[cols]
        burn_in = int(burn_in_fraction * chain_res.shape[0])
        chain_res = chain_res.iloc[burn_in:]  # type: ignore
        #print(f'discarded first {burn_in}')
        chains_list.append(chain_res)

    chain_res = pd.concat(chains_list)

    chain_res.columns = par_names
    ch_cons.add_chain(chain_res.values, parameters=list(
        chain_res.columns.values), name='MCMC',)

    ch_cons.configure(usetex=False, serif=False,
                      sigma2d=False, summary=True, bar_shade=True,
                      **config_kw)

    ch_cons_fig = ch_cons.plotter.plot(
        figsize=(figsize, figsize), filename=filename)
    #summary = ch_cons.plotter.plot_summary()

    #walks = ch_cons.plotter.plot_walks(chains=0, convolve=250)

    return chain_res, chains_list, ch_cons
